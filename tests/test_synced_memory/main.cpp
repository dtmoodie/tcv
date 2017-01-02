#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE "tcv_synced_memory"

#include <boost/thread.hpp>
#include <boost/test/unit_test.hpp>

#include <tcv/SyncedMemory.hpp>
#include <tcv/SyncedMemoryImpl.hpp>
#include <tcv/TypeTraits.hpp>
#include <tcv/DefaultAllocator.hpp>
#include <iostream>

using namespace tcv;
#ifdef CPU_ONLY
struct TestAllocator : public tcv::NoCudaAllocator
{
    typedef tcv::NoCudaAllocator Parent;
#else
struct TestAllocator : public tcv::DefaultAllocator
{
    typedef tcv::DefaultAllocator Parent;
#endif
    bool allocateGpu(void** ptr, size_t bytes)
    {
        Parent::allocateGpu(ptr, bytes);
        _gpu_allocation_count += bytes;
        return true;
    }
    bool allocateCpu(void** ptr, size_t bytes)
    {
        Parent::allocateCpu(ptr, bytes);
        _cpu_allocation_count += bytes;
        return true;
    }

    bool deallocateGpu(void* ptr, size_t bytes)
    {
        Parent::deallocateGpu(ptr, bytes);
        _gpu_allocation_count -= bytes;
        return true;
    }
    bool deallocateCpu(void* ptr, size_t bytes)
    {
        Parent::deallocateCpu(ptr, bytes);
        _cpu_allocation_count -= bytes;
        return true;
    }
    size_t _gpu_allocation_count = 0;
    size_t _cpu_allocation_count = 0;
};
BOOST_AUTO_TEST_CASE(setup)
{
    tcv::Allocator::setDefaultAllocator(new TestAllocator());
}
BOOST_AUTO_TEST_CASE(synced_memory_creation)
{
    SyncedMemory mem(100, DataType<float>::DType, nullptr);
    BOOST_REQUIRE_EQUAL(mem.getSize(), 100);
    BOOST_REQUIRE_EQUAL(mem.getElemSize(), sizeof(float));
    BOOST_REQUIRE_EQUAL(mem.getElemType(), DataType<float>::DType);
    const uint8_t* h_ptr = mem.getCpu();
    BOOST_REQUIRE(h_ptr);
#ifndef CPU_ONLY
    const uint8_t* d_ptr = mem.getGpu();
    BOOST_REQUIRE(d_ptr);
#endif

    float* data = (float*)mem.getCpuMutable();
    BOOST_REQUIRE(data);
    for(int i = 0; i < 25; ++i)
    {
        data[i] = i;
    }
    mem.resize(500);
    BOOST_REQUIRE_EQUAL(mem.getSize(), 500);
    data = (float*)mem.getCpuMutable();
    BOOST_REQUIRE(data);
    for (int i = 0; i < 125; ++i)
    {
        data[i] = i;
    }
}



BOOST_AUTO_TEST_CASE(custom_allocator)
{
    TestAllocator allocator;
    {
        SyncedMemory mem(100, DataType<float>::DType, &allocator);
        BOOST_REQUIRE_EQUAL(mem.getSize(), 100);
        BOOST_REQUIRE_EQUAL(mem.getElemSize(), sizeof(float));
        BOOST_REQUIRE_EQUAL(mem.getElemType(), DataType<float>::DType);
        const uint8_t* h_ptr = mem.getCpu();
        BOOST_REQUIRE(h_ptr);
        BOOST_REQUIRE_EQUAL(allocator._cpu_allocation_count, 100);
#ifndef CPU_ONLY
        const uint8_t* d_ptr = mem.getGpu();
        BOOST_REQUIRE(d_ptr);
        BOOST_REQUIRE_EQUAL(allocator._gpu_allocation_count, 100);
#endif
        float* data = (float*)mem.getCpuMutable();
        BOOST_REQUIRE(data);
        for (int i = 0; i < 25; ++i)
        {
            data[i] = i;
        }
        mem.resize(500);
        BOOST_REQUIRE_EQUAL(allocator._cpu_allocation_count, 0);
        BOOST_REQUIRE_EQUAL(allocator._gpu_allocation_count, 0);
        BOOST_REQUIRE_EQUAL(mem.getSize(), 500);
        data = (float*)mem.getCpuMutable();
        BOOST_REQUIRE_EQUAL(allocator._cpu_allocation_count, 500);
        BOOST_REQUIRE_EQUAL(allocator._gpu_allocation_count, 0);
        BOOST_REQUIRE(data);
        for (int i = 0; i < 125; ++i)
        {
            data[i] = i;
        }
#ifndef CPU_ONLY
        mem.getGpu();
        BOOST_REQUIRE_EQUAL(allocator._gpu_allocation_count, 500);
#endif
        BOOST_REQUIRE_EQUAL(allocator._cpu_allocation_count, 500);
    }
    BOOST_REQUIRE_EQUAL(allocator._cpu_allocation_count, 0);
    BOOST_REQUIRE_EQUAL(allocator._gpu_allocation_count, 0);
}

BOOST_AUTO_TEST_CASE(wrap_vec)
{
    TestAllocator allocator;
    std::vector<float> vector(1000);
    {
        SyncedMemory mem(vector, &allocator);
        BOOST_REQUIRE_EQUAL(mem.getSize(), 4000);
        BOOST_REQUIRE_EQUAL(mem.getElemSize(), sizeof(float));
        BOOST_REQUIRE_EQUAL(mem.getElemType(), DataType<float>::DType);
        const uint8_t* h_ptr = mem.getCpu();
        BOOST_REQUIRE_EQUAL((const float*)h_ptr, vector.data());
        BOOST_REQUIRE(h_ptr);
        BOOST_REQUIRE_EQUAL(allocator._cpu_allocation_count, 0);
#ifndef CPU_ONLY
        const uint8_t* d_ptr = mem.getGpu();
        BOOST_REQUIRE(d_ptr);
        BOOST_REQUIRE_EQUAL(allocator._gpu_allocation_count, 4000);
#endif
        float* data = (float*)mem.getCpuMutable();
        BOOST_REQUIRE(data);
        for (int i = 0; i < 1000; ++i)
        {
            data[i] = i;
        }
        for(int i = 0; i < 1000; ++i)
        {
            BOOST_REQUIRE_EQUAL(vector[i], i);
        }
        mem.resize(500);
        BOOST_REQUIRE_EQUAL(allocator._cpu_allocation_count, 0);
        BOOST_REQUIRE_EQUAL(allocator._gpu_allocation_count, 0);
        BOOST_REQUIRE_EQUAL(mem.getSize(), 500);
        data = (float*)mem.getCpuMutable();
        BOOST_REQUIRE_EQUAL(allocator._cpu_allocation_count, 500);
        BOOST_REQUIRE_EQUAL(allocator._gpu_allocation_count, 0);
        BOOST_REQUIRE(data);
        for (int i = 0; i < 125; ++i)
        {
            data[i] = i;
        }
#ifndef CPU_ONLY
        mem.getGpu();
        BOOST_REQUIRE_EQUAL(allocator._gpu_allocation_count, 500);
#endif
        BOOST_REQUIRE_EQUAL(allocator._cpu_allocation_count, 500);
    }
    BOOST_REQUIRE_EQUAL(allocator._cpu_allocation_count, 0);
    BOOST_REQUIRE_EQUAL(allocator._gpu_allocation_count, 0);
}

BOOST_AUTO_TEST_CASE(check_memory_leak)
{
    delete Allocator::getDefaultAllocator();
}