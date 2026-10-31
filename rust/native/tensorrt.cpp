// Copyright (c) 2026 Hailstorm contributors.
// Boundary adapted from Hailstorm's tensorrt_backend.py and Chatterbox Turbo
// (Copyright (c) 2025 Resemble AI, MIT License). Rust owns model policy.
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {
constexpr int kKvCount=48, kWidth=1024, kVocabulary=6563;
void fail(const std::string& text) { throw std::runtime_error(text); }
void report(char* out,size_t cap,const char* text) noexcept {
  if(out && cap) std::snprintf(out,cap,"%s",text);
}
void check(cudaError_t code,const char* operation) {
  if(code!=cudaSuccess) fail(std::string(operation)+": "+cudaGetErrorString(code));
}
struct Logger final:nvinfer1::ILogger {
  void log(Severity severity,const char* text) noexcept override {
    if(severity<=Severity::kWARNING) std::fprintf(stderr,"TensorRT: %s\n",text?text:"(no detail)");
  }
};
// TensorRT shares its logger across live runtimes. Token engines outlive a
// meanflow request, so every runtime must use one process-lifetime logger;
// otherwise registration can retain a pointer into a retired model owner.
Logger& process_logger() { static Logger logger;return logger; }
// Map serialized weights without retaining a second engine-sized host buffer.
struct Mapping {
  int fd=-1; size_t size=0; void* data=MAP_FAILED;
  explicit Mapping(const char* path) {
    fd=::open(path,O_RDONLY|O_CLOEXEC);
    if(fd<0) fail("could not open serialized TensorRT engine");
    struct stat status{};
    if(::fstat(fd,&status)!=0 || !S_ISREG(status.st_mode) || status.st_size<=0 ||
       static_cast<uintmax_t>(status.st_size)>std::numeric_limits<size_t>::max()) {
      ::close(fd); fd=-1; fail("TensorRT engine must be a non-empty regular file");
    }
    size=static_cast<size_t>(status.st_size);
    data=::mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0);
    if(data==MAP_FAILED) { ::close(fd);fd=-1;fail("could not map TensorRT engine"); }
  }
  ~Mapping() { if(data!=MAP_FAILED) ::munmap(data,size); if(fd>=0) ::close(fd); }
  Mapping(const Mapping&)=delete; Mapping& operator=(const Mapping&)=delete;
};
// Empty KV tensors still need a valid address; every device allocation has one owner.
struct Buffer {
  void* data=nullptr;
  explicit Buffer(size_t bytes=0) { check(cudaMalloc(&data,std::max<size_t>(bytes,1)),"cudaMalloc"); }
  ~Buffer() { if(data) cudaFree(data); }
  Buffer(Buffer&& other) noexcept:data(std::exchange(other.data,nullptr)){}
  Buffer& operator=(Buffer&& other) noexcept {
    if(this!=&other) { if(data) cudaFree(data); data=std::exchange(other.data,nullptr); } return *this;
  }
  Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
};
size_t volume(const nvinfer1::Dims& dims) {
  if(dims.nbDims<0) fail("unresolved TensorRT tensor rank");
  size_t count=1;
  for(int i=0;i<dims.nbDims;++i) {
    if(dims.d[i]<0) fail("unresolved TensorRT tensor dimension");
    const size_t dimension=static_cast<size_t>(dims.d[i]);
    if(dimension && count>std::numeric_limits<size_t>::max()/dimension) fail("TensorRT tensor size overflow");
    count*=dimension;
  }
  return count;
}
size_t width(nvinfer1::DataType type) {
  switch(type) {
    case nvinfer1::DataType::kHALF:return 2;
    case nvinfer1::DataType::kFLOAT:return 4;
    case nvinfer1::DataType::kINT32:return 4;
    case nvinfer1::DataType::kINT64:return 8;
    default:fail("unsupported TensorRT I/O dtype");
  }
  return 0;
}
size_t bytes(const nvinfer1::Dims& dims,nvinfer1::DataType type) {
  size_t count=volume(dims), element=width(type);
  if(count>std::numeric_limits<size_t>::max()/element) fail("TensorRT tensor byte size overflow");
  return count*element;
}
std::string kv(bool output,int index) {
  return std::string(output?"present_":"past_")+(index%2==0?"k_":"v_")+std::to_string(index/2);
}
nvinfer1::Dims shape(std::initializer_list<int64_t> values) {
  nvinfer1::Dims result{};result.nbDims=static_cast<int>(values.size());
  std::copy(values.begin(),values.end(),result.d);return result;
}
// Host conversion of final FP16 logits avoids a separate CUDA conversion kernel.
float to_float(uint16_t half) {
  uint32_t sign=static_cast<uint32_t>(half&0x8000)<<16;
  uint32_t exponent=(half>>10)&31, fraction=half&1023, bits;
  if(exponent==0) {
    if(fraction==0) bits=sign;
    else {
      int shift=0;while((fraction&1024)==0) {fraction<<=1;++shift;}
      bits=sign|(static_cast<uint32_t>(113-shift)<<23)|((fraction&1023)<<13);
    }
  } else if(exponent==31) bits=sign|0x7f800000U|(fraction<<13);
  else bits=sign|((exponent+112)<<23)|(fraction<<13);
  float result;std::memcpy(&result,&bits,sizeof(result));return result;
}
// Immutable weights/runtime outlive utterances. Contexts, KV and streams remain
// request-owned, so keeping the engine warm does not retain a previous prompt.
struct Engine {
  Logger& logger=process_logger();
  int device=0;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  explicit Engine(const char* path) {
    check(cudaGetDevice(&device),"cudaGetDevice");
    if(!initLibNvInferPlugins(&logger,"")) fail("TensorRT plugin initialisation failed");
    runtime.reset(nvinfer1::createInferRuntime(logger));
    if(!runtime) fail("TensorRT runtime creation failed");
    { Mapping mapped(path); engine.reset(runtime->deserializeCudaEngine(mapped.data,mapped.size)); }
    if(!engine) fail("TensorRT engine deserialisation failed");
    verify("embeds",nvinfer1::TensorIOMode::kINPUT,nvinfer1::DataType::kHALF);
    verify("position",nvinfer1::TensorIOMode::kINPUT,nvinfer1::DataType::kINT64);
    for(int i=0;i<kKvCount;++i) {
      verify(kv(false,i),nvinfer1::TensorIOMode::kINPUT,nvinfer1::DataType::kHALF);
      verify(kv(true,i),nvinfer1::TensorIOMode::kOUTPUT,nvinfer1::DataType::kHALF);
    }
    auto dtype=engine->getTensorDataType("logits");
    if(dtype!=nvinfer1::DataType::kHALF && dtype!=nvinfer1::DataType::kFLOAT) fail("logits must be FP16 or FP32");
    verify("logits",nvinfer1::TensorIOMode::kOUTPUT,dtype);
    int input_count=0;
    for(int i=0;i<engine->getNbIOTensors();++i) {
      const char* name=engine->getIOTensorName(i);
      if(engine->getTensorLocation(name)!=nvinfer1::TensorLocation::kDEVICE) fail(std::string("device I/O required: ")+name);
      if(engine->getTensorIOMode(name)==nvinfer1::TensorIOMode::kINPUT) ++input_count;
    }
    if(input_count!=2+kKvCount) fail("unexpected TensorRT engine inputs");
  }
  ~Engine() {
    // A later HTTP request/unload can execute on another blocking worker.
    // Reattach this engine's runtime primary context before native teardown.
    cudaSetDevice(device);
    engine.reset();runtime.reset();
  }
  void verify(const std::string& name,nvinfer1::TensorIOMode mode,nvinfer1::DataType dtype) {
    if(engine->getTensorIOMode(name.c_str())!=mode || engine->getTensorDataType(name.c_str())!=dtype)
      fail("TensorRT contract mismatch: "+name);
  }
};
struct Session {
  // Declared first so engine/runtime/logger retire after the execution context.
  std::shared_ptr<Engine> owner;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  cudaStream_t stream=nullptr;
  // Match the waveform owners' bounded workspace pattern, but keep token
  // scratch request-local: reset releases it before waveform work starts.
  struct Workspace {
    std::unique_ptr<Buffer> allocation;
    size_t capacity=0;
    void reserve(size_t bytes) {
      bytes=std::max<size_t>(bytes,1);
      if(bytes>capacity) {
        auto replacement=std::make_unique<Buffer>(bytes);
        allocation=std::move(replacement);capacity=bytes;
      }
    }
    void* data() const { return allocation?allocation->data:nullptr; }
    void clear() noexcept { allocation.reset();capacity=0; }
  };
  Workspace embeds,positions;
  // Separate input/output banks are essential: TensorRT concatenates the old
  // KV into present, so the two addresses must never alias during enqueue.
  std::array<Workspace,kKvCount> past,present;
  std::vector<Workspace> outputs;
  size_t length=0;
  std::mutex mutex;
  explicit Session(std::shared_ptr<Engine> weights):owner(std::move(weights)) {
    check(cudaSetDevice(owner->device),"cudaSetDevice");
    context.reset(owner->engine->createExecutionContext());
    if(!context) fail("TensorRT execution context creation failed");
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"cudaStreamCreateWithFlags");
  }
  ~Session() {
    cudaSetDevice(owner->device);
    if(stream) {
      cudaStreamSynchronize(stream);
      clear_scratch();context.reset();cudaStreamDestroy(stream);
    }
  }
  void input_shape(const std::string& name,const nvinfer1::Dims& dims) {
    if(!context->setInputShape(name.c_str(),dims)) fail("TensorRT rejected shape: "+name);
  }
  void bind(const std::string& name,void* address) {
    if(!context->setTensorAddress(name.c_str(),address)) fail("TensorRT rejected address: "+name);
  }
  int step(const uint16_t* input,size_t frames,float* logits,size_t capacity) {
    // A retained context may be checked out by a different blocking worker.
    check(cudaSetDevice(owner->device),"token worker device");
    if(!input || !logits) fail("embedding and logit pointers are required");
    if(capacity<kVocabulary) fail("logit destination needs capacity 6563");
    if(frames==0 || frames>512 || length>1024) fail("request exceeds retained TensorRT profile");
    embeds.reserve(frames*kWidth*sizeof(uint16_t));positions.reserve(frames*sizeof(int64_t));
    std::array<int64_t,512> host_positions{};
    for(size_t i=0;i<frames;++i) host_positions[i]=length+i;
    // Synchronous host copies keep input lifetimes safe on all error paths.
    check(cudaMemcpy(embeds.data(),input,frames*kWidth*sizeof(uint16_t),cudaMemcpyHostToDevice),"copy embeddings");
    check(cudaMemcpy(positions.data(),host_positions.data(),frames*sizeof(int64_t),cudaMemcpyHostToDevice),"copy positions");
    input_shape("embeds",shape({1,static_cast<int64_t>(frames),kWidth}));
    input_shape("position",shape({static_cast<int64_t>(frames)}));
    bind("embeds",embeds.data());bind("position",positions.data());
    for(int i=0;i<kKvCount;++i) {
      // Empty KV still requires a valid device address. Never grow a populated
      // input bank: reserve intentionally discards bytes, suitable for outputs.
      if(!length) past[i].reserve(1);
      auto name=kv(false,i);input_shape(name,shape({1,16,static_cast<int64_t>(length),64}));bind(name,past[i].data());
    }
    if(context->inferShapes(0,nullptr)!=0) fail("TensorRT dynamic shape inference failed");
    for(int i=0;i<kKvCount;++i) {
      auto name=kv(true,i);auto dims=context->getTensorShape(name.c_str());
      if(dims.nbDims!=4 || dims.d[0]!=1 || dims.d[1]!=16 || dims.d[2]!=static_cast<int64_t>(length+frames) || dims.d[3]!=64)
        fail("unexpected TensorRT KV shape: "+name);
      // Round only allocation capacity, never the logical shape or KV stride.
      // Each bank grows once per 128 frames instead of once per decoded token.
      const size_t reserved_frames=((length+frames+127)/128)*128;
      present[i].reserve(reserved_frames*16*64*sizeof(uint16_t));bind(name,present[i].data());
    }
    void* logit_address=nullptr;
    auto& engine=owner->engine;
    auto logit_type=engine->getTensorDataType("logits");auto logit_shape=context->getTensorShape("logits");
    if(logit_shape.nbDims!=3 || logit_shape.d[0]!=1 || logit_shape.d[1]!=static_cast<int64_t>(frames) || logit_shape.d[2]!=kVocabulary)
      fail("unexpected TensorRT logit shape");
    // Optional diagnostic outputs such as hidden must also have valid buffers.
    outputs.resize(engine->getNbIOTensors());
    for(int i=0;i<engine->getNbIOTensors();++i) {
      const char* name=engine->getIOTensorName(i);
      if(engine->getTensorIOMode(name)!=nvinfer1::TensorIOMode::kOUTPUT || std::strncmp(name,"present_",8)==0)continue;
      outputs[i].reserve(bytes(context->getTensorShape(name),engine->getTensorDataType(name)));
      bind(name,outputs[i].data());if(std::strcmp(name,"logits")==0)logit_address=outputs[i].data();
    }
    if(!logit_address) fail("TensorRT logits output missing");
    // enqueueV3 can submit work even on failure: drain before destroying buffers.
    bool enqueued=context->enqueueV3(stream);auto synchronized=cudaStreamSynchronize(stream);
    if(!enqueued)fail("TensorRT enqueueV3 failed");
    check(synchronized,"TensorRT inference stream");
    auto* row=static_cast<const char*>(logit_address)+(frames-1)*kVocabulary*width(logit_type);
    if(logit_type==nvinfer1::DataType::kFLOAT)check(cudaMemcpy(logits,row,kVocabulary*sizeof(float),cudaMemcpyDeviceToHost),"copy logits");
    else {
      std::array<uint16_t,kVocabulary> values{};
      check(cudaMemcpy(values.data(),row,values.size()*sizeof(uint16_t),cudaMemcpyDeviceToHost),"copy logits");
      for(size_t i=0;i<values.size();++i)logits[i]=to_float(values[i]);
    }
    // Commit cache only after inference and the host output copy succeed.
    past.swap(present);length+=frames;return kVocabulary;
  }
  void clear_scratch() noexcept {
    for(auto& b:past) b.clear();
    for(auto& b:present) b.clear();
    outputs.clear();embeds.clear();positions.clear();length=0;
  }
  void reset() noexcept {cudaSetDevice(owner->device);clear_scratch();}
};
#include "meanflow.inc"
#include "hift.inc"
#include "conformer.inc"
}
extern "C" {
void* cb_conformer_open(const char* path,char* error,size_t capacity) noexcept {
  report(error,capacity,"");
  try {if(!path||!*path)fail("Conformer engine path required");return new ConformerGraph(path);}
  catch(const std::exception& e){report(error,capacity,e.what());}
  catch(...){report(error,capacity,"unknown Conformer initialisation failure");}
  return nullptr;
}
int cb_conformer_run(void* handle,const float* input,size_t frames,float* output,
                     uint64_t* counts,char* error,size_t capacity) noexcept {
  report(error,capacity,"");
  try {
    if(!handle)fail("Conformer owner required");
    static_cast<ConformerGraph*>(handle)->execute(input,frames,output,counts);return 0;
  } catch(const std::exception& e){report(error,capacity,e.what());}
  catch(...){report(error,capacity,"unknown Conformer inference failure");}
  return -1;
}
void cb_conformer_close(void* handle) noexcept {delete static_cast<ConformerGraph*>(handle);}
void* cb_hift_open(const char* path,char* error,size_t capacity) noexcept {
  report(error,capacity,"");
  try {if(!path||!*path)fail("HiFT engine path required");return new HiftGraph(path);}
  catch(const std::exception& e){report(error,capacity,e.what());}
  catch(...){report(error,capacity,"unknown HiFT initialisation failure");}
  return nullptr;
}
int cb_hift_run(void* handle,const float* mel,const float* normal,const float* phases,
                const float* cache,size_t frames,size_t cached,float* pitch,float* source,
                float* waveform,uint64_t* counts,char* error,size_t capacity) noexcept {
  report(error,capacity,"");
  try {
    if(!handle)fail("HiFT owner required");
    static_cast<HiftGraph*>(handle)->execute(mel,normal,phases,cache,frames,cached,pitch,source,waveform,counts);
    return 0;
  } catch(const std::exception& e){report(error,capacity,e.what());}
  catch(...){report(error,capacity,"unknown HiFT inference failure");}
  return -1;
}
void cb_hift_close(void* handle) noexcept {delete static_cast<HiftGraph*>(handle);}
void* cb_meanflow_open(const char* path,char* error,size_t capacity) noexcept {
  report(error,capacity,"");
  try { if(!path||!*path)fail("meanflow engine path required");return new MeanflowGraph(path); }
  catch(const std::exception& e){report(error,capacity,e.what());}
  catch(...){report(error,capacity,"unknown meanflow initialisation failure");}
  return nullptr;
}
int cb_meanflow_run(void* handle,const float* x,const float* mask,const float* mu,
                    const float* spks,const float* cond,size_t frames,int steps,float t,float r,
                    float* out,uint64_t* counts,char* error,size_t capacity) noexcept {
  report(error,capacity,"");
  try {
    if(!handle)fail("meanflow owner required");
    static_cast<MeanflowGraph*>(handle)->execute(x,mask,mu,spks,cond,frames,steps,t,r,out,counts);
    return 0;
  } catch(const std::exception& e){report(error,capacity,e.what());}
  catch(...){report(error,capacity,"unknown meanflow inference failure");}
  return -1;
}
void cb_meanflow_close(void* handle) noexcept { delete static_cast<MeanflowGraph*>(handle); }
// Caller serialises handle lifetime: closing concurrently with any call is invalid.
void* cb_open(const char* path,char* error,size_t capacity) noexcept {
  report(error,capacity,"");
  try {if(!path || !*path)fail("TensorRT engine path required");return new Session(std::make_shared<Engine>(path));}
  catch(const std::exception& e){report(error,capacity,e.what());}
  catch(...){report(error,capacity,"unknown TensorRT initialisation failure");}
  return nullptr;
}
int cb_step(void* handle,const uint16_t* embeds,size_t frames,float* logits,size_t logits_capacity,char* error,size_t capacity) noexcept {
  report(error,capacity,"");
  try {if(!handle)fail("TensorRT session required");auto& session=*static_cast<Session*>(handle);std::lock_guard<std::mutex> lock(session.mutex);return session.step(embeds,frames,logits,logits_capacity);}
  catch(const std::exception& e){report(error,capacity,e.what());}
  catch(...){report(error,capacity,"unknown TensorRT inference failure");}
  return -1;
}
void cb_reset(void* handle) noexcept {
  if(!handle)return;
  try {auto& session=*static_cast<Session*>(handle);std::lock_guard<std::mutex> lock(session.mutex);session.reset();}catch(...){}
}
void cb_close(void* handle) noexcept {delete static_cast<Session*>(handle);}

// Separate, additive ABI for service-owned weights. Legacy CLI cb_open retains
// its original one-request lifetime. A session shares ownership if its cache is
// retired, so engine_close can never leave a dangling context.
void* cb_engine_open(const char* path,char* error,size_t capacity) noexcept {
  report(error,capacity,"");
  try {
    if(!path || !*path)fail("TensorRT engine path required");
    return new std::shared_ptr<Engine>(std::make_shared<Engine>(path));
  } catch(const std::exception& e){report(error,capacity,e.what());}
  catch(...){report(error,capacity,"unknown TensorRT engine initialisation failure");}
  return nullptr;
}
void* cb_session_open(void* handle,char* error,size_t capacity) noexcept {
  report(error,capacity,"");
  try {
    if(!handle)fail("TensorRT engine owner required");
    return new Session(*static_cast<std::shared_ptr<Engine>*>(handle));
  } catch(const std::exception& e){report(error,capacity,e.what());}
  catch(...){report(error,capacity,"unknown TensorRT context initialisation failure");}
  return nullptr;
}
void cb_engine_close(void* handle) noexcept {
  delete static_cast<std::shared_ptr<Engine>*>(handle);
}
}
