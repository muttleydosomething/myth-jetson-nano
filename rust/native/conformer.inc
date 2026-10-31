// Copyright (c) 2026 Hailstorm contributors.
// Complete Conformer graph, extending the existing bridge/workspace ownership.
// Only unpadded embeddings and final encoder features cross the host boundary.
struct ConformerGraph {
  Logger& logger=process_logger();
  int device=0;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  std::unique_ptr<MeanflowGraph::Stream> stream;
  std::array<MeanflowGraph::Workspace,2> buffers;
  const std::array<const char*,2> names={"input","features"};
  bool poisoned=false;
  explicit ConformerGraph(const char* path) {
    check(cudaGetDevice(&device),"Conformer device");
    if(!initLibNvInferPlugins(&logger,"")) fail("Conformer TensorRT plugin initialisation failed");
    runtime.reset(nvinfer1::createInferRuntime(logger));
    if(!runtime) fail("Conformer TensorRT runtime creation failed");
    { Mapping mapped(path);engine.reset(runtime->deserializeCudaEngine(mapped.data,mapped.size)); }
    if(!engine) fail("Conformer TensorRT engine deserialisation failed");
    if(engine->getNbIOTensors()!=2) fail("Conformer graph I/O count mismatch");
    for(size_t i=0;i<2;++i) {
      const char* name=names[i];
      if(engine->getTensorDataType(name)!=nvinfer1::DataType::kFLOAT ||
         engine->getTensorIOMode(name)!=(i==0?nvinfer1::TensorIOMode::kINPUT:nvinfer1::TensorIOMode::kOUTPUT) ||
         engine->getTensorLocation(name)!=nvinfer1::TensorLocation::kDEVICE ||
         engine->getTensorFormat(name)!=nvinfer1::TensorFormat::kLINEAR ||
         engine->isShapeInferenceIO(name))
        fail(std::string("Conformer graph I/O contract mismatch: ")+name);
    }
    context.reset(engine->createExecutionContext());
    if(!context) fail("Conformer execution context creation failed");
    stream=std::make_unique<MeanflowGraph::Stream>();
  }
  ~ConformerGraph() {
    cudaSetDevice(device);
    stream.reset(); // Drain queued work before releasing bound addresses.
    for(auto& b:buffers)b.clear();
    context.reset();engine.reset();runtime.reset();
  }
  void execute(const float* input,size_t frames,float* output,uint64_t* counts) {
    if(poisoned) fail("Conformer context retired after native inference failure; reopen model");
    if(!input||!output||!counts||!frames||frames>850) fail("invalid Conformer native arguments");
    check(cudaSetDevice(device),"Conformer worker device");
    // A failure at any stage retires this context. In particular no partial
    // submission survives the fence with request-owned host buffers dangling.
    try {
      if(!context->setInputShape("input",shape({1,static_cast<int64_t>(frames),512})))
        fail("Conformer frame length exceeds admitted TensorRT profile");
      if(context->inferShapes(0,nullptr)!=0) fail("Conformer dynamic shape inference failed");
      auto actual=context->getTensorShape("features");
      if(actual.nbDims!=3||actual.d[0]!=1||actual.d[1]!=static_cast<int64_t>(2*frames)||actual.d[2]!=512)
        fail("Conformer output shape mismatch");
      const std::array<size_t,2> sizes={frames*512*sizeof(float),2*frames*512*sizeof(float)};
      counts[0]=counts[1]=counts[2]=0;
      for(size_t i=0;i<2;++i) {
        buffers[i].reserve(sizes[i]);counts[2]+=buffers[i].capacity;
        if(!context->setTensorAddress(names[i],buffers[i].data)) fail("Conformer address rejected");
      }
      check(cudaMemcpyAsync(buffers[0].data,input,sizes[0],cudaMemcpyHostToDevice,stream->value),"Conformer upload");
      counts[0]=sizes[0];
      const bool enqueued=context->enqueueV3(stream->value);
      auto status=cudaStreamSynchronize(stream->value);
      if(!enqueued) fail("Conformer enqueueV3 failed");
      check(status,"Conformer inference stream");
      check(cudaMemcpyAsync(output,buffers[1].data,sizes[1],cudaMemcpyDeviceToHost,stream->value),"Conformer download");
      check(cudaStreamSynchronize(stream->value),"Conformer output fence");
      counts[1]=sizes[1];
    } catch(...) {cudaStreamSynchronize(stream->value);poisoned=true;throw;}
  }
};
