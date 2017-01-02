#pragma once
#include "SyncedMemory.hpp"
#include "TypeTraits.hpp"
#include "Defs.hpp"

namespace tcv
{
class Allocator;
class SyncedMemory;
class Tensor;
template<class T> class SyncedMemory_;
template<class T, int64_t... Sizes> class Tensor_;

class Tensor
{
public:

    template<class T, int64_t... Sizes>
    Tensor(Tensor_<T, Sizes...>& tensor):
        Tensor(static_cast<Tensor&>(tensor))
    {
        flags |= DataType<T>::DType << 12;
    }
    Tensor(Tensor& tensor);
    Tensor(Allocator* allocator = nullptr);
    ~Tensor();
    size_t numBytes() const;
    uint8_t elemType() const;

    const uint8_t* getCpu(cudaStream_t stream) const;
    uint8_t* getCpuMutable(cudaStream_t stream);

    const uint8_t* getGpu(cudaStream_t stream) const;
    uint8_t* getGpuMutable(cudaStream_t stream);

    size_t getStride(int dim = 0);
    size_t getStrideBytes(int dim = 0);

    size_t getNumBytes(int dim = 0);

    size_t getNumElements(int dim = 0);
    size_t getShape(int dim = 0);
    static size_t getStaticShape(int dim = 0);
    size_t getShapeBytes(int dim = 0);
    size_t getNumDims();
    size_t getSize(int dim = 0);

    template<class T, int64_t... Sizes>
    Tensor& operator=(Tensor_<T, Sizes...>& tensor)
    {
        this->operator=(static_cast<Tensor&>(tensor));
        flags |= DataType<T>::DType << 12;
        return *this;
    }
    Tensor& operator=(Tensor& tensor);

    SyncedMemory* data;
    SyncedMemory_<size_t>* stride;
    SyncedMemory_<size_t>* shape;
    int dims;
    Allocator* allocator;
    uint64_t flags;
    int* refCount;
    void cleanup();
    void decrement();
    void resize(size_t size);
};

} // namespace tcv
