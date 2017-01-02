#include "tcv/Allocator.hpp"
#include "tcv/DefaultAllocator.hpp"
#include "tcv/SyncedMemory.hpp"
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
    
    thread_local Allocator* t_allocator = nullptr;
    static Allocator* g_allocator = Allocator::getStandardAllocator();

    Allocator* Allocator::getStandardAllocator()
    {
        static DefaultAllocator* g_allocator = nullptr;
        if(g_allocator == nullptr)
        {
            g_allocator = new DefaultAllocator();
        }
        return g_allocator;
    }
    Allocator* Allocator::getDefaultAllocator()
    {
        if(t_allocator)
        {
            return t_allocator;
        }
        return g_allocator;
    }
    void Allocator::setDefaultAllocator(Allocator* allocator)
    {
        g_allocator = allocator;
    }
    void Allocator::setThreadAllocator(Allocator* allocator)
    {
        t_allocator = allocator;
    }
}