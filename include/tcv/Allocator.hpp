#pragma once
#include <cstddef>
#include <stdint.h>
namespace tcv
{
    class SyncedMemory;
    class Tensor;

    class Allocator
    {
    public:
        virtual ~Allocator() {}
        virtual bool allocateGpu(void** ptr, size_t bytes) = 0;
        virtual bool allocateCpu(void** ptr, size_t bytes) = 0;
        virtual bool allocate(SyncedMemory* synced_mem, size_t bytes, int elemSize) = 0;
        virtual bool allocate(Tensor* tensor, size_t bytes, int elemSize) = 0;

        virtual bool deallocateGpu(void* ptr, size_t bytes) = 0;
        virtual bool deallocateCpu(void* ptr, size_t bytes) = 0;
        virtual bool deallocate(SyncedMemory* synced_mem) = 0;
        virtual bool deallocate(Tensor* tensor) = 0;
        static Allocator* getDefaultAllocator();
    protected:
        void setCpu(SyncedMemory* mem, uint8_t* ptr);
        void setGpu(SyncedMemory* mem, uint8_t* ptr);
        uint8_t* getCpu(SyncedMemory* mem);
        uint8_t* getGpu(SyncedMemory* mem);
    };
}
