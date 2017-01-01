#pragma once
#include "tcv/TypeTraits.hpp"
namespace tcv
{
class SyncedMemory;
template<class T> class SyncedMemory_;

template<class T>
SyncedMemory::SyncedMemory(std::vector<T>& vec, Allocator* allocator):
    SyncedMemory(CpuPtr<uint8_t>((uint8_t*)vec.data()), vec.size() * sizeof(T), DataType<T>::DType, allocator)
{
}

template<class T>
SyncedMemory_<T>::SyncedMemory_(Allocator* allocator):
    SyncedMemory(allocator)
{

}

template<class T>
SyncedMemory_<T>::SyncedMemory_(size_t elements, Allocator* allocator):
    SyncedMemory(sizeof(T)*elements, allocator)
{

}

template<class T>
SyncedMemory_<T>::~SyncedMemory_()
{

}

template<class T>
bool SyncedMemory_<T>::resize(size_t elements)
{
    return SyncedMemory::resize(elements*sizeof(T));
}

template<class T>
size_t  SyncedMemory_<T>::getSize() const
{
    return SyncedMemory::getSize() / sizeof(T);
}

template<class T>
uint8_t SyncedMemory_<T>::getGpuId() const
{
    return SyncedMemory::getGpuId();
}

template<class T>
SyncedMemory::Flags   SyncedMemory_<T>::getSyncState() const
{
    return SyncedMemory::getSyncState();
}

template<class T>
void SyncedMemory_<T>::setCpuData(T* ptr, size_t num_elements)
{
    SyncedMemory::setCpuData((uint8_t*)ptr, num_elements*sizeof(T));
}


template<class T>
void SyncedMemory_<T>::setGpuData(T* ptr, size_t num_elements)
{
    SyncedMemory::setGpuData((uint8_t*)ptr, num_elements*sizeof(T));
}

template<class T>
const T* SyncedMemory_<T>::getCpu(cudaStream_t stream)
{
    return (const T*)SyncedMemory::getCpu(stream);
}

template<class T>
const T* SyncedMemory_<T>::getCpu(size_t offset, size_t size,
                                  cudaStream_t stream)
{
    return (const T*)SyncedMemory::getCpu(offset*sizeof(T), size*sizeof(T), stream);
}

template<class T>
const T* SyncedMemory_<T>::getCpu(size_t offset, size_t width,
                                  size_t height, size_t stride,
                                  cudaStream_t stream)
{
    return (const T*)SyncedMemory::getCpu(offset*sizeof(T), width*sizeof(T), height, stride, stream);
}

template<class T>
T* SyncedMemory_<T>::getCpuMutable(cudaStream_t stream)
{
    return (T*)SyncedMemory::getCpu(stream);
}

template<class T>
T* SyncedMemory_<T>::getCpuMutable(size_t offset, size_t size,
                                   cudaStream_t stream)
{
    return (T*)SyncedMemory::getCpuMutable(offset*sizeof(T), size*sizeof(T), stream);
}

template<class T>
T* SyncedMemory_<T>::getCpuMutable(size_t offset, size_t width,
                                   size_t height, size_t stride,
                                   cudaStream_t stream)
{
    return (T*)SyncedMemory::getCpuMutable(offset*sizeof(T), width*sizeof(T), height, stride, stream);
}

// Request a chunk of data, this will set dirty flags on sections of requested data
template<class T>
const T* SyncedMemory_<T>::getGpu(cudaStream_t stream)
{
    return (const T*)getGpu(stream);
}

template<class T>
const T* SyncedMemory_<T>::getGpu(size_t offset, size_t size,
    cudaStream_t stream)
{
    return (const T*)getGpu(offset*sizeof(T), size*sizeof(T), stream);
}

template<class T>
const T* SyncedMemory_<T>::getGpu(size_t offset, size_t width,
                                  size_t height, size_t stride,
                                  cudaStream_t stream)
{
    return (const T*)getGpu(offset*sizeof(T), width*sizeof(T), height, stride, stream);
}

template<class T>
T& SyncedMemory_<T>::operator[](size_t index)
{
    return reinterpret_cast<T*>(this->_cpu_data)[index];
}

template<class T>
T*SyncedMemory_<T>::getGpuMutable(cudaStream_t stream)
{
    return (T*)SyncedMemory::getGpuMutable(stream);
}

template<class T>
T* SyncedMemory_<T>::getGpuMutable(size_t offset, size_t size,
                                   cudaStream_t stream)
{
    return (T*)SyncedMemory::getGpuMutable(offset*sizeof(T), size*sizeof(T), stream);
}

template<class T>
T* SyncedMemory_<T>::getGpuMutable(size_t offset, size_t width,
                                   size_t height, size_t stride,
                                   cudaStream_t stream)
{
    return (T*)SyncedMemory::getGpuMutable(offset*sizeof(T), width*sizeof(T), height, stride, stream);
}

template<class T>
void SyncedMemory_<T>::copyTo(SyncedMemory& other, cudaStream_t stream)
{
    SyncedMemory::copyTo(other, stream);
}
template<class T>
void SyncedMemory_<T>::copyFromCpu(const void* data, size_t size, cudaStream_t stream)
{
    SyncedMemory::copyFromCpu(data, size, stream);
}
template<class T>
void SyncedMemory_<T>::copyFromGpu(const void* data, size_t size, cudaStream_t stream)
{
    SyncedMemory::copyFromGpu(data, size, stream);
}
template<class T>
SyncedMemory_<T>* SyncedMemory_<T>::clone(cudaStream_t stream)
{
    synchronize(stream);
    SyncedMemory_<T>* output = new SyncedMemory_<T>();
    output->_size = this->_size;
    output->_allocator = this->_allocator;
    output->_flags = this->_flags;
    if (_gpu_data)
    {
        auto ptr = output->getGpuMutable(stream);
        cudaMemcpyAsync(ptr, _gpu_data, _size, cudaMemcpyDeviceToDevice, stream);
        return output;
    } else if(_cpu_data)
    {
        auto ptr = output->getCpuMutable(stream);
        cudaMemcpyAsync(ptr, _cpu_data, _size, cudaMemcpyHostToHost, stream);
    }
    return output;
}
}
