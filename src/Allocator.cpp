#include "tcv/Tensor.hpp"
#include <cuda_runtime_api.h>
#include <cstdlib>
namespace tcv
{
    class DefaultAllocator: public Tensor::Allocator
    {
    public:
        bool Allocate(Tensor* tensor, size_t bytes, int elemSize)
        {
            void* ptr = nullptr;
            cudaMallocHost(&ptr, bytes);
            if(ptr)
            {
                tensor->refCount = (int*)malloc(sizeof(int));
                *tensor->refCount = 1;
                tensor->h_data = (uint8_t*)ptr;
                tensor->allocator = this;
                return true;
            }
            return false;
        }
        bool Deallocate(Tensor* tensor)
        {
            cudaFreeHost(tensor->h_data);
            free(tensor->refCount);
            return true;
        }
    };
    static DefaultAllocator g_allocator;
    Tensor::Allocator* Tensor::Allocator::getDefaultAllocator()
    {
        return &g_allocator;
    }
}
