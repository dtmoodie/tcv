#include "tcv/Tensor.hpp"
#include "tcv/Allocator.hpp"

#include <cuda_runtime_api.h>

using namespace tcv;
SyncedMemory::DirtyBlock::DirtyBlock(uint8_t* p, size_t w, size_t s, size_t h, bool cpu)
    : ptr(p), stride(s), width(w), height(h), dirty_cpu(cpu)
{
}
SyncedMemory::SyncedMemory(Allocator* allocator)
    : _allocator(allocator)
    , _cpu_data(nullptr)
    , _gpu_data(nullptr)
    , _flags(0)
    , _size(0)
{
    
}
SyncedMemory::SyncedMemory(size_t size, Allocator* allocator)
    : _allocator(allocator)
    , _cpu_data(nullptr)
    , _gpu_data(nullptr)
    , _flags(0)
    , _size(size)
{

}

SyncedMemory::~SyncedMemory()
{

}

bool SyncedMemory::resize(size_t size)
{
    if(_cpu_data)
    {
        if(!_allocator->deallocateCpu(_cpu_data, _size))
        {
        
        }   Allocator::getDefaultAllocator()->deallocateCpu(_cpu_data, _size);
        _cpu_data = nullptr;
    }
    if(_gpu_data)
    {
        if(!_allocator->deallocateGpu(_gpu_data, _size))
        {
            Allocator::getDefaultAllocator()->deallocateGpu(_gpu_data, _size);
        }
        _gpu_data = nullptr;
    }
    _size = size;
    return true;
}
size_t  SyncedMemory::getSize() const
{
    return _size;
}

uint8_t SyncedMemory::getGpuId() const
{
    return _flags & 0xFF;
}

SyncedMemory::Flags SyncedMemory::getSyncState() const
{
    if(_dirty_blocks.size() == 0)
    {
        return Synchronized_e;
    }
}

void SyncedMemory::setCpuData(void* ptr, size_t size)
{
    
}

void SyncedMemory::setGpuData(void* ptr, size_t size)
{

}

// This will dirty the entire data block
uint8_t* SyncedMemory::getCpuMutable(cudaStream_t stream)
{
    for (auto itr = _dirty_blocks.begin(); itr != _dirty_blocks.end(); ++itr)
    {
        if(itr->dirty_cpu == false)
        {
            cudaMemcpy2DAsync(_cpu_data + (itr->ptr - _gpu_data), itr->stride, 
                              itr->ptr, itr->stride, itr->width, itr->height, 
                              cudaMemcpyDeviceToHost, stream);   
        }
    }
    _dirty_blocks.clear();
    _dirty_blocks.emplace_back(_cpu_data, _size, _size, 1, true);
    return _cpu_data;
}

uint8_t* SyncedMemory::getGpuMutable(cudaStream_t stream)
{
    for (auto itr = _dirty_blocks.begin(); itr != _dirty_blocks.end(); ++itr)
    {
        if (itr->dirty_cpu == true)
        {
            cudaMemcpy2DAsync(_gpu_data + (itr->ptr - _cpu_data), itr->stride,
                itr->ptr, itr->stride, itr->width, itr->height,
                cudaMemcpyHostToDevice, stream);
        }
    }
    _dirty_blocks.clear();
    _dirty_blocks.emplace_back(_gpu_data, _size, _size, 1, false);
    return _gpu_data;
}

const uint8_t* SyncedMemory::getCpu(cudaStream_t stream)
{
    if(_cpu_data == nullptr)
    {
        allocateCpu();
    }
    for (auto itr = _dirty_blocks.begin(); itr != _dirty_blocks.end(); ++itr)
    {
        if (itr->dirty_cpu == false)
        {
            cudaMemcpy2DAsync(_cpu_data + (itr->ptr - _gpu_data), itr->stride,
                itr->ptr, itr->stride, itr->width, itr->height,
                cudaMemcpyDeviceToHost, stream);
        }
    }
    _dirty_blocks.clear();
    return _cpu_data;
}

const uint8_t* SyncedMemory::getGpu(cudaStream_t stream)
{
    if(_cpu_data == nullptr)
    {
        allocateGpu();
    }
    for (auto itr = _dirty_blocks.begin(); itr != _dirty_blocks.end(); ++itr)
    {
        if (itr->dirty_cpu == true)
        {
            cudaMemcpy2DAsync(_gpu_data + (itr->ptr - _cpu_data), itr->stride,
                itr->ptr, itr->stride, itr->width, itr->height,
                cudaMemcpyHostToDevice, stream);
        }
    }
    _dirty_blocks.clear();
    return _gpu_data;
}

// Request a chunk of data, this will set dirty flags on sections of requested data
uint8_t* SyncedMemory::getCpuMutable(size_t offset, size_t size, cudaStream_t stream)
{
    return nullptr;
}

uint8_t* SyncedMemory::getGpuMutable(size_t offset, size_t size, cudaStream_t stream)
{
    return nullptr;
}
// This just lets you grab a chunk of data, will only synchronize what is needed
const uint8_t* SyncedMemory::getCpu(size_t offset, size_t size, cudaStream_t stream)
{
    return nullptr;
}

const uint8_t* SyncedMemory::getGpu(size_t offset, size_t size, cudaStream_t stream )
{
    return nullptr;
}

void SyncedMemory::synchronize(cudaStream_t stream)
{

}

void SyncedMemory::allocateCpu()
{
    if(_allocator)
    {
        if(!this->_allocator->allocateCpu((void**)&_cpu_data, _size))
        {
            Allocator::getDefaultAllocator()->allocateCpu((void**)&_cpu_data, _size);
            return;
        }
    }
    Allocator::getDefaultAllocator()->allocateCpu((void**)&_cpu_data, _size);
}

void SyncedMemory::allocateGpu()
{
    if(_allocator)
    {
        if(!this->_allocator->allocateGpu((void**)&_gpu_data, _size))
        {
            Allocator::getDefaultAllocator()->allocateGpu((void**)&_gpu_data, _size);
            return;
        }
    }
    Allocator::getDefaultAllocator()->allocateGpu((void**)&_gpu_data, _size);
}
