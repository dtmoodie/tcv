#include "tcv/MemoryLeakDebugAllocator.hpp"
#include "yar/Logging.hpp"
#include <iostream>

using namespace tcv;
MemoryLeakDebugAllocator::~MemoryLeakDebugAllocator()
{
    if(_cpu_allocations.size())
    {
        std::cout << "\n\n =============== " << _cpu_allocations.size() << " CPU memory leaks ================\n\n";
    }
    for(auto& leak : _cpu_allocations)
    {
        std::cout << "======================\n";
        std::cout << leak.second.size << "\n";
        std::cout << leak.second.callstack << std::endl;
    }
    if (_gpu_allocations.size())
    {
        std::cout << "\n\n =============== " << _gpu_allocations.size() << " GPU memory leaks ================\n\n";
    }
    for (auto& leak : _gpu_allocations)
    {
        std::cout << "======================\n";
        std::cout << leak.second.size << "\n";
        std::cout << leak.second.callstack << std::endl;
    }
}
bool MemoryLeakDebugAllocator::allocateGpu(void** ptr, size_t bytes)
{
    if(DefaultAllocator::allocateGpu(ptr, bytes))
    {
        _gpu_allocations[*ptr] = Allocation(yar::print_callstack(3, true), bytes);
        return true;
    }
    return false;
}
bool MemoryLeakDebugAllocator::allocateCpu(void** ptr, size_t bytes)
{
    if(DefaultAllocator::allocateCpu(ptr, bytes))
    {
        _cpu_allocations[*ptr] = Allocation(yar::print_callstack(3, true), bytes);
        return true;
    }
    return false;
}
bool MemoryLeakDebugAllocator::allocate(SyncedMemory* synced_mem, size_t bytes, uint8_t elemType)
{
    if(DefaultAllocator::allocate(synced_mem, bytes, elemType))
    {

        return true;
    }
    return false;
}
bool MemoryLeakDebugAllocator::allocate(Tensor* tensor, size_t bytes, uint8_t elemType)
{
    if(DefaultAllocator::allocate(tensor, bytes, elemType))
    {

        return true;
    }
    return false;
}

bool MemoryLeakDebugAllocator::deallocateGpu(void* ptr, size_t bytes)
{
    if(DefaultAllocator::deallocateGpu(ptr, bytes))
    {
        _gpu_allocations.erase(ptr);
        return true;
    }
    return false;
}
bool MemoryLeakDebugAllocator::deallocateCpu(void* ptr, size_t bytes)
{
    if(DefaultAllocator::deallocateCpu(ptr, bytes))
    {
        _cpu_allocations.erase(ptr);
        return true;
    }
    return false;
}
bool MemoryLeakDebugAllocator::deallocate(SyncedMemory* synced_mem)
{
    if(DefaultAllocator::deallocate(synced_mem))
    {
        return true;
    }
    return false;
}
bool MemoryLeakDebugAllocator::deallocate(Tensor* tensor)
{
    if(DefaultAllocator::deallocate(tensor))
    {

        return true;
    }
    return false;
}