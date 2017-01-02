#pragma once

#include "Allocator.hpp"

namespace tcv
{
    class DefaultAllocator : public Allocator
    {
    public:
        bool allocateGpu(void** ptr, size_t bytes);
        bool allocateCpu(void** ptr, size_t bytes);
        bool allocate(SyncedMemory* synced_mem, size_t bytes, uint8_t elemType);
        bool allocate(Tensor* tensor, size_t bytes, uint8_t elemType);

        bool deallocateGpu(void* ptr, size_t bytes);
        bool deallocateCpu(void* ptr, size_t bytes);
        bool deallocate(SyncedMemory* synced_mem);
        bool deallocate(Tensor* tensor);
    };

    class NoCudaAllocator:public Allocator
    {
    public:
        bool allocateGpu(void** ptr, size_t bytes);
        bool allocateCpu(void** ptr, size_t bytes);
        bool allocate(SyncedMemory* synced_mem, size_t bytes, uint8_t elemType);
        bool allocate(Tensor* tensor, size_t bytes, uint8_t elemType);

        bool deallocateGpu(void* ptr, size_t bytes);
        bool deallocateCpu(void* ptr, size_t bytes);
        bool deallocate(SyncedMemory* synced_mem);
        bool deallocate(Tensor* tensor);
    };
}