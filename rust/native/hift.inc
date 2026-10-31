// Copyright (c) 2026 Hailstorm contributors.
// Included in the existing TensorRT bridge. One owner retains the full HiFT
// graph; only request inputs and final pitch/source/waveform cross the ABI.
struct HiftGraph {
  Logger& logger=process_logger();
  int device=0;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  // Reuse the established stream/workspace owners and their failure fences.
  std::unique_ptr<MeanflowGraph::Stream> stream;
  std::array<MeanflowGraph::Workspace,8> buffers;
  const std::array<const char*,8> names={"mel","normal","phases","cache_source","cache_samples","pitch","source","waveform"};
  bool poisoned=false;
  explicit HiftGraph(const char* path) {
    check(cudaGetDevice(&device),"HiFT device");
    if(!initLibNvInferPlugins(&logger,"")) fail("HiFT TensorRT plugin initialisation failed");
    runtime.reset(nvinfer1::createInferRuntime(logger));
    if(!runtime) fail("HiFT TensorRT runtime creation failed");
    { Mapping mapped(path);engine.reset(runtime->deserializeCudaEngine(mapped.data,mapped.size)); }
    if(!engine) fail("HiFT TensorRT engine deserialisation failed");
    if(engine->getNbIOTensors()!=static_cast<int>(names.size())) fail("HiFT graph I/O count mismatch");
    for(size_t i=0;i<names.size();++i) {
      const char* name=names[i];
      if(engine->getTensorDataType(name)!=(i==4?nvinfer1::DataType::kINT32:nvinfer1::DataType::kFLOAT) ||
         engine->getTensorIOMode(name)!=(i<5?nvinfer1::TensorIOMode::kINPUT:nvinfer1::TensorIOMode::kOUTPUT) ||
         engine->getTensorLocation(name)!=nvinfer1::TensorLocation::kDEVICE ||
         engine->getTensorFormat(name)!=nvinfer1::TensorFormat::kLINEAR ||
         engine->isShapeInferenceIO(name))
        fail(std::string("HiFT graph I/O contract mismatch: ")+name);
    }
    context.reset(engine->createExecutionContext());
    if(!context) fail("HiFT execution context creation failed");
    stream=std::make_unique<MeanflowGraph::Stream>();
  }
  ~HiftGraph() {
    cudaSetDevice(device);
    stream.reset(); // Drain before releasing any input or activation buffer.
    for(auto& b:buffers) b.clear();
    context.reset();engine.reset();runtime.reset();
  }
  void execute(const float* mel,const float* normal,const float* phases,const float* cache,
               size_t frames,size_t cached,float* pitch,float* source,float* waveform,uint64_t* counts) {
    if(poisoned) fail("HiFT context retired after native inference failure; reopen model");
    if(!mel||!normal||!phases||!pitch||!source||!waveform||!counts||!frames||
       frames>INT32_MAX/480||cached>frames*480||(cached&&!cache))
      fail("invalid HiFT native arguments");
    check(cudaSetDevice(device),"HiFT worker device");
    const size_t samples=frames*480;
    const std::array<size_t,8> sizes={frames*80*4,samples*9*4,9*4,samples*4,4,frames*4,samples*4,samples*4};
    const std::array<nvinfer1::Dims,5> dims={shape({1,80,static_cast<int64_t>(frames)}),
      shape({1,9,static_cast<int64_t>(samples)}),shape({1,9,1}),
      shape({1,1,static_cast<int64_t>(samples)}),shape({1})};
    for(size_t i=0;i<5;++i)
      if(!context->setInputShape(names[i],dims[i])) fail("HiFT frame length exceeds admitted TensorRT profile");
    if(context->inferShapes(0,nullptr)!=0) fail("HiFT dynamic shape inference failed");
    const std::array<nvinfer1::Dims,3> outputs={shape({1,static_cast<int64_t>(frames)}),
      shape({1,1,static_cast<int64_t>(samples)}),shape({1,static_cast<int64_t>(samples)})};
    for(size_t i=5;i<8;++i) {
      auto actual=context->getTensorShape(names[i]);auto expected=outputs[i-5];
      if(actual.nbDims!=expected.nbDims) fail("HiFT output rank mismatch");
      for(int j=0;j<actual.nbDims;++j)
        if(actual.d[j]!=expected.d[j]) fail("HiFT output shape mismatch");
    }
    counts[0]=counts[1]=counts[2]=0;
    for(size_t i=0;i<8;++i) {
      buffers[i].reserve(sizes[i]);counts[2]+=buffers[i].capacity;
      if(!context->setTensorAddress(names[i],buffers[i].data)) fail("HiFT TensorRT address rejected");
    }
    const float* inputs[]={mel,normal,phases};
    int32_t count=static_cast<int32_t>(cached);
    try {
      for(size_t i=0;i<3;++i) {
        check(cudaMemcpyAsync(buffers[i].data,inputs[i],sizes[i],cudaMemcpyHostToDevice,stream->value),"HiFT upload");
        counts[0]+=sizes[i];
      }
      // Prefix content belongs to this request only. Clear the carrier even
      // when the prefix shrinks, so no previous request data remains readable.
      check(cudaMemsetAsync(buffers[3].data,0,sizes[3],stream->value),"HiFT prefix clear");
      if(cached) check(cudaMemcpyAsync(buffers[3].data,cache,cached*4,cudaMemcpyHostToDevice,stream->value),"HiFT prefix upload");
      check(cudaMemcpyAsync(buffers[4].data,&count,4,cudaMemcpyHostToDevice,stream->value),"HiFT prefix length");
      counts[0]+=cached*4+4;
      const bool enqueued=context->enqueueV3(stream->value);
      auto status=cudaStreamSynchronize(stream->value);
      if(!enqueued) fail("HiFT enqueueV3 failed");
      check(status,"HiFT inference stream");
      float* out[]={pitch,source,waveform};
      for(size_t i=5;i<8;++i) {
        check(cudaMemcpyAsync(out[i-5],buffers[i].data,sizes[i],cudaMemcpyDeviceToHost,stream->value),"HiFT download");
        counts[1]+=sizes[i];
      }
      check(cudaStreamSynchronize(stream->value),"HiFT output fence");
    } catch(...) {cudaStreamSynchronize(stream->value);poisoned=true;throw;}
  }
};
