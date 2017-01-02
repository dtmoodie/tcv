#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <list>
#include <cstdint>
#include <cstddef>
#include <vector>

#define DISALLOW_COPY_MOVE_AND_ASSIGN(TypeName) \
TypeName(const TypeName&);                      \
void operator=(const TypeName&);                \
TypeName(TypeName&&);                           \
void operator=(const TypeName&&)

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace tcv
{
class Allocator;

template<class T> struct CpuPtr
{
    CpuPtr(T* ptr_):
        ptr(ptr_)
    {
    }
    T* ptr;
};

template<class T> struct GpuPtr
{
    GpuPtr(T* ptr_):
        ptr(ptr_)
    {
    }
    T* ptr;
};

// This class manages synchronization of memory between the CPU and the GPU.
// This class knowns nothing of the shape of the data
class SyncedMemory
{
public:
    enum Flags : uint32_t
    {
        // This first 8 bits ares reserved for the device id of the gpu
        // If this flag is thrown, then operations will not add dirty blocks
        // This should be used when wrapping other synced memory classes that do
        // their own synchronization
        HeadAtCpu_e = 256,
        HeadAtGpu_e = 257,
        Synchronized_e = 258,
        DoNotSync_e = 259,
        OwnsCpu_e = 512,
        OwnsGpu_e = 1024,
        ReadOnlyCpu_e = 2048,
        ReadOnlyGpu_e = 4096
        // The next bits 4 bits are used to flag element info
    };

    SyncedMemory(Allocator* allocator = nullptr);
    SyncedMemory(size_t size, uint8_t elemType, Allocator* allocator = nullptr);
    explicit SyncedMemory(CpuPtr<uint8_t> data, size_t size, uint8_t elemType, Allocator* allocator = nullptr);
    explicit SyncedMemory(GpuPtr<uint8_t> data, size_t size, uint8_t elemType, Allocator* allocator = nullptr);
    template<class T>
    SyncedMemory(std::vector<T>& vec, Allocator* allocator = nullptr);

    ~SyncedMemory();

    bool resize(size_t size);
    size_t getSize() const;
    uint8_t getElemSize() const;
    uint8_t getElemType() const;
    uint8_t getGpuId() const;
    Flags   getSyncState() const;

    void setCpuData(uint8_t* ptr, size_t size, uint16_t flags = 0);
    void setGpuData(uint8_t* ptr, size_t size, uint16_t flags = 0);

    const uint8_t* getCpu(cudaStream_t stream = 0);
    const uint8_t* getCpu(size_t offset, size_t size,
                          cudaStream_t stream = 0);
    const uint8_t* getCpu(size_t offset, size_t width,
        size_t height, size_t stride,
                          cudaStream_t stream = 0);

    uint8_t* getCpuMutable(cudaStream_t stream = 0);
    uint8_t* getCpuMutable(size_t offset, size_t size,
                           cudaStream_t stream = 0);
    uint8_t* getCpuMutable(size_t offset, size_t width,
        size_t height, size_t stride,
                           cudaStream_t stream = 0);

    // Request a chunk of data, this will set dirty flags on sections of requested data
    const uint8_t* getGpu(cudaStream_t stream = 0);
    const uint8_t* getGpu(size_t offset, size_t size,
                          cudaStream_t stream = 0);
    const uint8_t* getGpu(size_t offset, size_t width,
        size_t height, size_t stride,
                          cudaStream_t stream = 0);

    uint8_t* getGpuMutable(cudaStream_t stream = 0);
    uint8_t* getGpuMutable(size_t offset, size_t size,
                           cudaStream_t stream = 0);
    uint8_t* getGpuMutable(size_t offset, size_t width,
        size_t height, size_t stride,
                           cudaStream_t stream = 0);

    void synchronize(cudaStream_t stream = 0);
    SyncedMemory* clone(cudaStream_t stream = 0);
    void copyTo(SyncedMemory& other, cudaStream_t stream = 0);
    void copyFromCpu(const void* data, size_t size, cudaStream_t stream = 0);
    void copyFromGpu(const void* data, size_t size, cudaStream_t stream = 0);
protected:
    void allocateCpu();
    void allocateGpu();
    DISALLOW_COPY_MOVE_AND_ASSIGN(SyncedMemory);
    friend class Allocator;
    struct DirtyBlock
    {
        DirtyBlock(uint8_t* p, size_t w, size_t s, size_t h, bool cpu);
        // Pointer to beginning of dirty data
        uint8_t*  ptr;
        // Size of data that was modified
        size_t width;
        size_t stride;
        size_t height;
        // If true, the CPU memory chunk was modified, else gpu
        bool   dirty_cpu;
    };
    uint8_t*              _cpu_data;
    uint8_t*              _gpu_data;
    uint32_t               _flags;
    size_t                _size;
    Allocator*            _allocator;
    std::list<DirtyBlock> _dirty_blocks;
};

template<class T> class SyncedMemory_: protected SyncedMemory
{
public:
    SyncedMemory_(Allocator* allocator = nullptr);
    SyncedMemory_(size_t elements, Allocator* allocator = nullptr);
    ~SyncedMemory_();

    bool resize(size_t elements);
    size_t  getSize() const;
    uint8_t getGpuId() const;
    Flags   getSyncState() const;

    void setCpuData(T* ptr, size_t num_elements);
    void setGpuData(T* ptr, size_t num_elements);

    const T* getCpu(cudaStream_t stream = 0);
    const T* getCpu(size_t offset, size_t size,
                    cudaStream_t stream = 0);
    const T* getCpu(size_t offset, size_t width,
        size_t height, size_t stride,
                    cudaStream_t stream = 0);

    T* getCpuMutable(cudaStream_t stream = 0);
    T* getCpuMutable(size_t offset, size_t size, cudaStream_t stream = 0);
    T* getCpuMutable(size_t offset, size_t width,
        size_t height, size_t stride,
                     cudaStream_t stream = 0);

    // Request a chunk of data, this will set dirty flags on sections of requested data
    const T* getGpu(cudaStream_t stream = 0);
    const T* getGpu(size_t offset, size_t size,
                    cudaStream_t stream = 0);
    const T* getGpu(size_t offset, size_t width,
        size_t height, size_t stride,
                    cudaStream_t stream = 0);

    T& operator[](size_t index);

    T* getGpuMutable(cudaStream_t stream = 0);
    T* getGpuMutable(size_t offset, size_t size,
                     cudaStream_t stream = 0);
    T* getGpuMutable(size_t offset, size_t width,
        size_t height, size_t stride,
                     cudaStream_t stream = 0);
    SyncedMemory_<T>* clone(cudaStream_t stream = 0);
    void copyTo(SyncedMemory& other, cudaStream_t stream = 0);
    void copyFromCpu(const void* data, size_t size, cudaStream_t stream = 0);
    void copyFromGpu(const void* data, size_t size, cudaStream_t stream = 0);
protected:
    DISALLOW_COPY_MOVE_AND_ASSIGN(SyncedMemory_);

};

}
