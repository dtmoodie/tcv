#include "tcv/Tensor.hpp"
#include "tcv/Allocator.hpp"
#include "tcv/SyncedMemoryImpl.hpp"
namespace tcv
{

Tensor::Tensor(Allocator* allocator):
    shape(new SyncedMemory_<size_t>(allocator)),
    stride(new SyncedMemory_<size_t>(allocator))
{
    data = nullptr;
    dims = 0;
    if(allocator == nullptr)
    {
        this->allocator = Allocator::getDefaultAllocator();
    }else
    {
        this->allocator = allocator;
    }
    flags = 0;
    refCount = 0;
}
Tensor::Tensor(Tensor& tensor)
{
    data = tensor.data;
    stride = tensor.stride->clone();
    shape = tensor.shape->clone();
    dims = tensor.dims;
    allocator = tensor.allocator;
    flags = tensor.flags;
    refCount = tensor.refCount;
    if (refCount)
    {
        ++(*refCount);
    }
}
Tensor& Tensor::operator=(Tensor& tensor)
{
    decrement();
    data = tensor.data;
    stride = tensor.stride->clone();
    shape = tensor.shape->clone();
    dims = tensor.dims;
    allocator = tensor.allocator;
    flags = tensor.flags;
    refCount = tensor.refCount;
    if (refCount)
    {
        ++(*refCount);
    }
    return *this;
}
Tensor::~Tensor()
{
    decrement();
    if(stride)
        delete stride;
    if(shape)
        delete shape;
}

void Tensor::decrement()
{
    if (refCount)
    {
        --*refCount;
        if (*refCount == 0)
        {
            if (allocator)
            {
                if (!allocator->deallocate(this))
                {
                    Allocator::getDefaultAllocator()->deallocate(this);
                }
            }else
            {
                Allocator::getDefaultAllocator()->deallocate(this);
            }
        }
    }
}

size_t Tensor::numBytes() const
{
    size_t size = 1;
    for (int i = 0; i < dims; ++i)
    {
        size *= (*shape)[i];
    }
    return size;
}

uint8_t Tensor::elemType() const
{
    return (flags >> 12) & 0x0F;
}

void Tensor::cleanup()
{
    if (allocator)
    {
        allocator->deallocate(this);
    }
}

const uint8_t* Tensor::getCpu(cudaStream_t stream) const
{
    return data->getCpu(stream);
}

uint8_t* Tensor::getCpuMutable(cudaStream_t stream)
{
    return data->getCpuMutable(stream);
}

const uint8_t* Tensor::getGpu(cudaStream_t stream) const
{
    return data->getCpu(stream);
}

uint8_t* Tensor::getGpuMutable(cudaStream_t stream)
{
    return data->getGpuMutable(stream);
}

size_t Tensor::getStride(int dim)
{
    return getStrideBytes(dim) / data->getElemSize();
}

size_t Tensor::getStrideBytes(int dim)
{
    return (*stride)[dim];
}

size_t Tensor::getNumBytes(int dim)
{
    size_t bytes = 1;
    size_t size = data->getElemSize();
    for(int i = dim; i < dims; ++i)
    {
        bytes *= (*shape)[i] / size;
    }
    return bytes * size;
}

size_t Tensor::getNumElements(int dim)
{
    return getNumBytes(dim) / data->getElemSize();
}

size_t Tensor::getShape(int dim)
{
    return (*shape)[dim] / data->getElemSize();
}

size_t Tensor::getStaticShape(int dim)
{
    return 0;
}

size_t Tensor::getShapeBytes(int dim)
{
    return (*shape)[dim];
}

size_t Tensor::getNumDims()
{
    return dims;
}

size_t Tensor::getSize(int dim)
{
    size_t size = 1;
    for(int i = dim; i < getNumDims(); ++i)
    {
        size *= getShape(i);
    }
    return size;
}

void Tensor::resize(size_t size)
{
    if(data)
    {
        if(size != data->getSize())
        {
            decrement();
            data = nullptr;
            refCount = nullptr;
        }
    }
    
    if (allocator)
    {
        allocator->allocate(this, size, elemType());
    }
    else
    {
        tcv::Allocator::getDefaultAllocator()->allocate(this, size, elemType());
    }
}
}