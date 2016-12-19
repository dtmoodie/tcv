#include "tcv/Allocator.hpp"
#include "tcv/Tensor.hpp"
#include "tcv/SyncedMemory.hpp"
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

    class DefaultAllocator: public Allocator
    {
    public:
        bool allocateGpu(void** ptr, size_t bytes)
        {
            cudaMalloc(ptr, bytes);
            return true;
        }
        bool allocateCpu(void** ptr, size_t bytes)
        {
            *ptr = malloc(bytes);
            return *ptr != nullptr;
        }
        bool allocate(SyncedMemory* synced_mem, size_t bytes, int elemSize)
        {
            return true;
        }
        bool allocate(Tensor* tensor, size_t bytes, int elemSize)
        {
            void* ptr = nullptr;
            cudaMallocHost(&ptr, bytes);
            if (ptr)
            {
                tensor->refCount = (int*)malloc(sizeof(int));
                *tensor->refCount = 1;
                tensor->data = new SyncedMemory(bytes);
                tensor->allocator = this;
                return true;
            }
            return false;
        }

        bool deallocateGpu(void* ptr, size_t bytes)
        {
            return true;
        }
        bool deallocateCpu(void* ptr, size_t bytes)
        {
            return true;
        }
        bool deallocate(SyncedMemory* synced_mem)
        {
            return true;
        }
        bool deallocate(Tensor* tensor)
        {
            if(deallocate(tensor->data))
            {
                free(tensor->refCount);
                return true;
            }
            return false;
        }
    };
    static DefaultAllocator g_allocator;
    Allocator* Allocator::getDefaultAllocator()
    {
        return &g_allocator;
    }
}
