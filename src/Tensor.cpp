#include "tcv/Tensor.hpp"
#include "tcv/Allocator.hpp"
namespace tcv
{

Tensor::Tensor()
{
    data = nullptr;
    stride = nullptr;
    shape = nullptr;
    dims = 0;
    allocator = nullptr;
    flags = 0;
}

Tensor::~Tensor()
{
    if (refCount)
    {
        --*refCount;
        if (*refCount == 0)
        {
            if (allocator)
            {
                if (!allocator->deallocate(this))
                {
                    Allocator::getDefaultAllocator()->deallocate(this);
                }
            }
        }
    }
}

void Tensor::cleanup()
{
    if (allocator)
    {
        allocator->deallocate(this);
    }
}

}