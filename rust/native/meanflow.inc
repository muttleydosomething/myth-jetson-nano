// Copyright (c) 2026 Hailstorm contributors.
// Included inside tensorrt.cpp's anonymous namespace: share its checked CUDA,
// mapping and FP32 I/O helpers, without a second runtime/bridge implementation.
// One owner retains weights, context, stream and bounded reusable device inputs.
struct MeanflowGraph {
  Logger& logger=process_logger();
  int device=0;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  struct Stream {
    cudaStream_t value=nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking),"meanflow stream"); }
    ~Stream() { if(value) { cudaStreamSynchronize(value);cudaStreamDestroy(value); } }
  };
  std::unique_ptr<Stream> stream;
  struct Workspace {
    void* data=nullptr;
    size_t capacity=0;
    ~Workspace() { clear(); }
    void clear() { if(data) cudaFree(data);data=nullptr;capacity=0; }
    void reserve(size_t n) {
      if(n>capacity) {
        Buffer replacement(n);
        clear();data=std::exchange(replacement.data,nullptr);capacity=n;
      }
    }
  };
  // x and next_x ping-pong between Euler steps. Derivative is downloaded only
  // by the numerical oracle; ordinary decoding downloads one final mel.
  std::array<Workspace,9> buffers;
  const std::array<const char*,9> names={"x","mask","mu","spks","cond","t","r","next_x","derivative"};
  bool poisoned=false;
  explicit MeanflowGraph(const char* path) {
    check(cudaGetDevice(&device),"meanflow device");
    if(!initLibNvInferPlugins(&logger,"")) fail("meanflow TensorRT plugin initialisation failed");
    runtime.reset(nvinfer1::createInferRuntime(logger));
    if(!runtime) fail("meanflow TensorRT runtime creation failed");
    { Mapping mapped(path);engine.reset(runtime->deserializeCudaEngine(mapped.data,mapped.size)); }
    if(!engine) fail("meanflow TensorRT engine deserialisation failed");
    if(engine->getNbIOTensors()!=static_cast<int>(names.size())) fail("meanflow graph I/O count mismatch");
    for(size_t i=0;i<names.size();++i) {
      const char* name=names[i];
      if(engine->getTensorDataType(name)!=nvinfer1::DataType::kFLOAT ||
         engine->getTensorIOMode(name)!=(i<7?nvinfer1::TensorIOMode::kINPUT:nvinfer1::TensorIOMode::kOUTPUT) ||
         engine->getTensorLocation(name)!=nvinfer1::TensorLocation::kDEVICE ||
         engine->getTensorFormat(name)!=nvinfer1::TensorFormat::kLINEAR ||
         engine->isShapeInferenceIO(name))
        fail(std::string("meanflow graph I/O contract mismatch: ")+name);
    }
    context.reset(engine->createExecutionContext());
    if(!context) fail("meanflow execution context creation failed");
    stream=std::make_unique<Stream>();
  }
  ~MeanflowGraph() {
    cudaSetDevice(device);
    // Drain before freeing any input, output or activation workspace.
    stream.reset();
    for(auto& b:buffers) b.clear();
    context.reset();engine.reset();runtime.reset();
  }
  void bind(size_t index,void* address) {
    if(!context->setTensorAddress(names[index],address)) fail("meanflow TensorRT address rejected");
  }
  void input_shape(size_t index,const nvinfer1::Dims& dims) {
    if(!context->setInputShape(names[index],dims)) fail("meanflow frame length exceeds admitted TensorRT profile");
  }
  void execute(const float* x,const float* mask,const float* mu,const float* spks,const float* cond,
               size_t frames,int steps,float t,float r,float* out,uint64_t* counts) {
    if(poisoned) fail("meanflow context retired after native inference failure; reopen model");
    if(!x||!mask||!mu||!spks||!cond||!out||!counts||!frames||frames>INT32_MAX||steps<0)
      fail("invalid meanflow native arguments");
    check(cudaSetDevice(device),"meanflow worker device");
    const size_t mel=frames*80*sizeof(float);
    const std::array<size_t,9> sizes={mel,frames*sizeof(float),mel,80*sizeof(float),mel,4,4,mel,mel};
    for(size_t i=0;i<7;++i)
      input_shape(i,i==3?shape({1,80}):i>=5?shape({1}):shape({1,i==1?1:80,static_cast<int64_t>(frames)}));
    if(context->inferShapes(0,nullptr)!=0) fail("meanflow dynamic shape inference failed");
    for(size_t i=7;i<9;++i) {
      auto dims=context->getTensorShape(names[i]);
      if(dims.nbDims!=3||dims.d[0]!=1||dims.d[1]!=80||dims.d[2]!=static_cast<int64_t>(frames))
        fail("meanflow output shape mismatch");
    }
    for(size_t i=0;i<9;++i) { buffers[i].reserve(sizes[i]);bind(i,buffers[i].data); }
    counts[0]=counts[1]=counts[2]=0;
    for(const auto& b:buffers) counts[2]+=b.capacity;
    const float* inputs[]={x,mask,mu,spks,cond};
    void* current=buffers[0].data;
    void* next=buffers[7].data;
    // Time sources outlive the error-path fence too. Pageable host copies can
    // stage asynchronously: all uploads must share the nonblocking graph stream.
    float times[2]={t,r};
    try {
      for(size_t i=0;i<5;++i) {
        check(cudaMemcpyAsync(buffers[i].data,inputs[i],sizes[i],cudaMemcpyHostToDevice,stream->value),"meanflow upload");
        counts[0]+=sizes[i];
      }
      const int iterations=steps?steps:1;
      for(int i=0;i<iterations;++i) {
        times[0]=steps?static_cast<float>(i)/steps:t;
        times[1]=steps?static_cast<float>(i+1)/steps:r;
        for(size_t j=0;j<2;++j) {
          check(cudaMemcpyAsync(buffers[5+j].data,&times[j],4,cudaMemcpyHostToDevice,stream->value),"meanflow time upload");
          counts[0]+=4;
        }
        bind(0,current);bind(7,next);
        const bool enqueued=context->enqueueV3(stream->value);
        // Always drain even when enqueue fails: work may already be submitted.
        auto status=cudaStreamSynchronize(stream->value);
        if(!enqueued) fail("meanflow enqueueV3 failed");
        check(status,"meanflow inference stream");
        std::swap(current,next);
      }
      check(cudaMemcpy(out,steps?current:buffers[8].data,mel,cudaMemcpyDeviceToHost),"meanflow final download");
      counts[1]=mel;
    } catch(...) {
      cudaStreamSynchronize(stream->value);poisoned=true;throw;
    }
  }
};
