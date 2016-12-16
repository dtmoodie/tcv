#include "tcv/Tensor.hpp"

namespace tcv
{
    class DefaultAllocator: public Tensor::Allocator
    {
    public:
        virtual bool Allocate(Tensor* tensor, size_t bytes, int elemSize)
        {
            void* ptr = malloc(bytes);
            tensor->h_data = (uint8_t*)ptr;
            return ptr != 0;
        }
        virtual bool Deallocate(Tensor* tensor)
        {
            free(tensor->h_data);
            return true;
        }
        
    };
    static DefaultAllocator g_allocator;
    Tensor::Allocator* Tensor::Allocator::getDefaultAllocator()
    {
        return &g_allocator;
    }
}