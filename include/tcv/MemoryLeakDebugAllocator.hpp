#pragma once
#include "DefaultAllocator.hpp"
#include <map>

namespace tcv
{
    class MemoryLeakDebugAllocator : public DefaultAllocator
    {
    public:
        ~MemoryLeakDebugAllocator();
        bool allocateGpu(void** ptr, size_t bytes);
        bool allocateCpu(void** ptr, size_t bytes);
        bool allocate(SyncedMemory* synced_mem, size_t bytes, uint8_t elemType);
        bool allocate(Tensor* tensor, size_t bytes, uint8_t elemType);

        bool deallocateGpu(void* ptr, size_t bytes);
        bool deallocateCpu(void* ptr, size_t bytes);
        bool deallocate(SyncedMemory* synced_mem);
        bool deallocate(Tensor* tensor);
    private:
        struct Allocation
        {
            Allocation(std::string&& cs = "", size_t size_ = 0):
                callstack(cs), size(size_){}

            std::string callstack;
            size_t size;
        };
        std::map<void*, Allocation> _cpu_allocations;
        std::map<void*, Allocation> _gpu_allocations;
    };
}