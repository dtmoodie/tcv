#pragma once
namespace tcv
{
class SyncedMemory;
template<class T> class SyncedMemory_;
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
    return SyncedMemory::resize(elements);
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

}
