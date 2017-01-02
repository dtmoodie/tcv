#include "tcv/DefaultAllocator.hpp"
#include <cuda_runtime_api.h>
#include "tcv/Tensor.hpp"
#include "tcv/SyncedMemory.hpp"
using namespace tcv;

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
bool DefaultAllocator::allocate(SyncedMemory* synced_mem, size_t bytes, uint8_t elemType)
{
    return true;
}
bool DefaultAllocator::allocate(Tensor* tensor, size_t bytes, uint8_t elemType)
{
    if (tensor->data)
    {
        delete tensor->data;
        tensor->data = nullptr;
    }
    tensor->refCount = (int*)malloc(sizeof(int));
    *tensor->refCount = 1;
    tensor->data = new SyncedMemory(bytes, elemType, this);
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
    if (deallocate(tensor->data))
    {
        tensor->data = nullptr;
        free(tensor->refCount);
        tensor->refCount = nullptr;
        return true;
    }
    return false;
}