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

SyncedMemory::SyncedMemory(size_t size, uint8_t elemType, Allocator* allocator)
    : _allocator(allocator)
    , _cpu_data(nullptr)
    , _gpu_data(nullptr)
    , _flags(0)
    , _size(size)
{
    _flags |= (int)elemType << 12;
    if(_allocator == nullptr)
    {
        _allocator = Allocator::getDefaultAllocator();
    }

}

SyncedMemory::SyncedMemory(CpuPtr<uint8_t> data, size_t size, uint8_t elemType, Allocator* allocator):
    _allocator(allocator),
    _cpu_data((uint8_t*)data.ptr),
    _gpu_data(nullptr),
    _size(size),
    _flags(0)
{
    _flags = (int)elemType << 12;
    if (_allocator == nullptr)
    {
        _allocator = Allocator::getDefaultAllocator();
    }
}

SyncedMemory::SyncedMemory(GpuPtr<uint8_t> data, size_t size, uint8_t elemType, Allocator* allocator):
    _allocator(allocator),
    _cpu_data(nullptr),
    _gpu_data((uint8_t*)data.ptr),
    _size(size),
    _flags((int)elemType << 12)
{
    if (_allocator == nullptr)
    {
        _allocator = Allocator::getDefaultAllocator();
    }
}

SyncedMemory::~SyncedMemory()
{
    if(_cpu_data && _flags & OwnsCpu_e)
    {
        if(_allocator)
        {
            if (!_allocator->deallocateCpu(_cpu_data, _size))
                Allocator::getDefaultAllocator()->deallocateCpu(_cpu_data, _size);
        }
        else
        {
            Allocator::getDefaultAllocator()->deallocateCpu(_cpu_data, _size);
        }
    }
    if(_gpu_data && _flags & OwnsGpu_e)
    {
        if(_allocator)
        {
            if(!_allocator->deallocateGpu(_gpu_data, _size))
                Allocator::getDefaultAllocator()->deallocateGpu(_gpu_data, _size);
        }else
        {
            Allocator::getDefaultAllocator()->deallocateGpu(_gpu_data, _size);
        }
    }
}

bool SyncedMemory::resize(size_t size)
{
    if(size == _size)
        return true;
    if(_cpu_data)
    {
        if(_flags & OwnsCpu_e)
        {
            if (_allocator)
            {
                if (!_allocator->deallocateCpu(_cpu_data, _size))
                {
                    Allocator::getDefaultAllocator()->deallocateCpu(_cpu_data, _size);
                }
            }
            else
            {
                Allocator::getDefaultAllocator()->deallocateCpu(_cpu_data, _size);
            }
        }
        _cpu_data = nullptr;
    }
    if(_gpu_data)
    {
        if(_flags & OwnsGpu_e)
        {
            if (_allocator)
            {
                if (!_allocator->deallocateGpu(_gpu_data, _size))
                {
                    Allocator::getDefaultAllocator()->deallocateGpu(_gpu_data, _size);
                }
            }
            else
            {
                Allocator::getDefaultAllocator()->deallocateGpu(_gpu_data, _size);
            }
        }
        _gpu_data = nullptr;
    }
    _dirty_blocks.clear();
    _size = size;
    return true;
}

size_t SyncedMemory::getSize() const
{
    return _size;
}

uint8_t SyncedMemory::getElemSize() const
{
    return typeToSize(getElemType());
}

uint8_t SyncedMemory::getElemType() const
{
    return (_flags >> 12) & 0x0F;
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

void SyncedMemory::setCpuData(uint8_t* ptr, size_t size, uint16_t flags)
{
    _cpu_data = ptr;
    _size = size;
    _flags |= flags;
}

void SyncedMemory::setGpuData(uint8_t* ptr, size_t size, uint16_t flags)
{
    _gpu_data = ptr;
    _size = size;
    _flags |= flags;
}

// This will dirty the entire data block
uint8_t* SyncedMemory::getCpuMutable(cudaStream_t stream)
{
    if (_cpu_data == nullptr)
    {
        allocateCpu();
    }
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
    if (_gpu_data == nullptr)
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
    if(_gpu_data == nullptr)
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
        if(this->_allocator->allocateCpu((void**)&_cpu_data, _size))
        { 
            _flags |= OwnsCpu_e;
            return;
        }
    }
    Allocator::getDefaultAllocator()->allocateCpu((void**)&_cpu_data, _size);
    _flags |= OwnsCpu_e;
}

void SyncedMemory::allocateGpu()
{
    if(_allocator)
    {
        if(this->_allocator->allocateGpu((void**)&_gpu_data, _size))
        {
            _flags |= OwnsGpu_e;
            return;
        }
    }
    Allocator::getDefaultAllocator()->allocateGpu((void**)&_gpu_data, _size);
    _flags |= OwnsGpu_e;
}
SyncedMemory* SyncedMemory::clone(cudaStream_t stream)
{
    SyncedMemory* output = new SyncedMemory(_size, getElemType(), _allocator);
    copyTo(*output, stream);
    return output;
}
void SyncedMemory::copyTo(SyncedMemory& other, cudaStream_t stream)
{
    // Need to figoure out dirtying
    synchronize(stream);
    cudaMemcpyAsync(other.getGpuMutable(), getGpu(), _size, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(other.getCpuMutable(), getCpu(), _size, cudaMemcpyHostToHost, stream);
    other._dirty_blocks.clear();
}
void SyncedMemory::copyFromCpu(const void* data, size_t size, cudaStream_t stream)
{
    if(_size != size)
        resize(size);
    cudaMemcpyAsync(getCpuMutable(stream), data, size, cudaMemcpyHostToHost, stream);
}
void SyncedMemory::copyFromGpu(const void* data, size_t size, cudaStream_t stream)
{
    if (_size != size)
        resize(size);
    cudaMemcpyAsync(getGpuMutable(stream), data, size, cudaMemcpyDeviceToDevice, stream);
}
