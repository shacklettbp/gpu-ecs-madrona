#pragma once

#include <madrona/macros.hpp>
#include <madrona/span.hpp>
#include <madrona/optional.hpp>

#include <array>

#ifndef MADRONA_PYTHON_VISIBILITY
#define MADRONA_PYTHON_VISIBILITY MADRONA_IMPORT
#endif

#ifdef MADRONA_CUDA_SUPPORT
#include <cuda_runtime.h>
#endif

namespace madrona {
namespace py {

#ifdef MADRONA_CUDA_SUPPORT
class MADRONA_PYTHON_VISIBILITY CudaSync final {
public:
    CudaSync(cudaExternalSemaphore_t sema);
    void wait(uint64_t strm);

private:
#ifdef MADRONA_LINUX
    // These classes have to be virtual on linux so a unique typeinfo
    // is emitted. Otherwise every user of this class gets a weak symbol
    // reference and nanobind can't map the types correctly
    virtual void key_();
#endif

    cudaExternalSemaphore_t sema_;
};
#endif

class MADRONA_PYTHON_VISIBILITY Tensor final {
public:
    enum class ElementType {
        UInt8,
        Int8,
        Int16,
        Int32,
        Int64,
        Float16,
        Float32,
    };

    Tensor(void *dev_ptr, ElementType type,
           Span<const int64_t> dimensions,
           Optional<int> gpu_id);
    
    inline void * devicePtr() const { return dev_ptr_; }
    inline ElementType type() const { return type_; }
    inline bool isOnGPU() const { return gpu_id_ != -1; }
    inline int gpuID() const { return gpu_id_; }
    inline int64_t numDims() const { return num_dimensions_; }
    inline const int64_t *dims() const { return dimensions_.data(); }

    static inline constexpr int64_t maxDimensions = 16;
private:
#ifdef MADRONA_LINUX
    virtual void key_();
#endif

    void *dev_ptr_;
    ElementType type_;
    int gpu_id_;

    int64_t num_dimensions_;
    std::array<int64_t, maxDimensions> dimensions_;
};

}
}
