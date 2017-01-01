#include "tcv/Allocator.hpp"
#include "tcv/Tensor.hpp"
#include "tcv/SyncedMemory.hpp"
#include "tcv/DefaultAllocator.hpp"
#include <cuda_runtime_api.h>
#include <cstdlib>
namespace tcv
{
    void Allocator::setCpu(SyncedMemory* mem, uint8_t* ptr)
    {
        if(mem->_cpu_data)
        {
            this->deallocateCpu(mem->_cpu_data, mem->_size);
        }
        mem->_cpu_data = ptr;
    }
    void Allocator::setGpu(SyncedMemory* mem, uint8_t* ptr)
    {
        if (mem->_gpu_data)
        {
            this->deallocateGpu(mem->_gpu_data, mem->_size);
        }
        mem->_gpu_data = ptr;
    }
    uint8_t* Allocator::getCpu(SyncedMemory* mem)
    {
        return mem->_cpu_data;
    }
    uint8_t* Allocator::getGpu(SyncedMemory* mem)
    {
        return mem->_gpu_data;
    }

    
    bool DefaultAllocator::allocateGpu(void** ptr, size_t bytes)
    {
        cudaMalloc(ptr, bytes);
        return true;
    }
    bool DefaultAllocator::allocateCpu(void** ptr, size_t bytes)
    {
        cudaMallocHost(ptr, bytes);
        return *ptr != nullptr;
    }
    bool DefaultAllocator::allocate(SyncedMemory* synced_mem, size_t bytes, uint8_t elemSize)
    {
        return true;
    }
    bool DefaultAllocator::allocate(Tensor* tensor, size_t bytes, uint8_t elemSize)
    {
        if(tensor->data)
        {
            delete tensor->data;
            tensor->data = nullptr;
        }
        tensor->refCount = (int*)malloc(sizeof(int));
        *tensor->refCount = 1;
        tensor->data = new SyncedMemory(bytes, elemSize, this);
        tensor->allocator = this;
        return true;
    }

    bool DefaultAllocator::deallocateGpu(void* ptr, size_t bytes)
    {
        cudaFree(ptr);
        return true;
    }
    bool DefaultAllocator::deallocateCpu(void* ptr, size_t bytes)
    {
        cudaFreeHost(ptr);
        return true;
    }
    bool DefaultAllocator::deallocate(SyncedMemory* synced_mem)
    {
        delete synced_mem;
        return true;
    }
    bool DefaultAllocator::deallocate(Tensor* tensor)
    {
        if(deallocate(tensor->data))
        {
            tensor->data = nullptr;
            free(tensor->refCount);
            tensor->refCount = nullptr;
            return true;
        }
        return false;
    }
    static DefaultAllocator g_allocator;
    Allocator* Allocator::getDefaultAllocator()
    {
        return &g_allocator;
    }
}
