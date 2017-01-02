#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE "tcv_tensor"
#include "tcv/MemoryLeakDebugAllocator.hpp"
#include <boost/thread.hpp>
#include <boost/test/unit_test.hpp>

#include <tcv/Tensor.hpp>
#include <tcv/TTensor.hpp>
#include <tcv/SyncedMemory.hpp>
#include <tcv/SyncedMemoryImpl.hpp>
#include <tcv/View.hpp>
#ifdef WITH_OPENCV
#include <tcv/cvTypeTraits.hpp>
#endif
#include <iostream>
using namespace tcv;

#ifdef WITH_OPENCV
typedef cv::Vec3f TestType;
#else
typedef float3 TestType;
#endif
#ifdef CPU_ONLY
    typedef NoCudaAllocator TestAllocator;
#else
    typedef MemoryLeakDebugAllocator TestAllocator;
#endif
BOOST_AUTO_TEST_CASE(static_shape_tensor)
{
    {
        ScopedAllocator<TestAllocator> allocator;
    
    Tensor_<float, 4, 5, 3> Mat4x5x3f;
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getStrideBytes(), 5*3*sizeof(float));
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getStrideBytes(1), 3 * sizeof(float));
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getStrideBytes(2), sizeof(float));
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getNumBytes(0), 4*5*3*sizeof(float));
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getNumBytes(1), 3*5 * sizeof(float));
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getNumBytes(2), 3*sizeof(float));
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getShape(0), 4);
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getShape(1), 5);
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getShape(2), 3);
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getShapeBytes(0), 4 * sizeof(float));
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getShapeBytes(1), 5 * sizeof(float));
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getShapeBytes(2), 3 * sizeof(float));
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getNumElements(0), 4 * 5 * 3);
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getNumElements(1), 5 * 3);
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getNumElements(2), 3);
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.shapeFlags(), STATIC_SHAPE + STATIC_DIMS);
    for(int i = 0; i < Mat4x5x3f.getShape(0); ++i)
    {
        for(int j = 0; j < Mat4x5x3f.getShape(1); ++j)
        {
            for(int k = 0; k < Mat4x5x3f.getShape(2); ++k)
            {
                BOOST_REQUIRE_EQUAL(Mat4x5x3f.getOffset(i, j, k), k + j * 3 + i * 5*3);
            }
        }
    }
    
    int num_elems = Mat4x5x3f.getNumElements(0);
    const float* data = Mat4x5x3f.getCpu();
    BOOST_REQUIRE(data);
    for(int i = 0; i < Mat4x5x3f.getShape(0); ++i)
    {
        for(int j = 0; j < Mat4x5x3f.getShape(1); ++j)
        {
            for(int k = 0; k < Mat4x5x3f.getShape(2); ++k)
            {
                Mat4x5x3f(i,j,k) = 100 * i + 10 * j + k;
            }
        }
    }
    std::cout << "\noperator() access\n";
    for (int i = 0; i < Mat4x5x3f.getShape(0); ++i)
    {
        for (int j = 0; j < Mat4x5x3f.getShape(1); ++j)
        {
            for (int k = 0; k < Mat4x5x3f.getShape(2); ++k)
            {
                std::cout << std::setw(3) << std::setfill('0') << Mat4x5x3f(i, j, k) << " ";
            }
        }
        std::cout << std::endl;
    }
    std::cout << "\nraw memory access\n";

    for (int i = 0; i < Mat4x5x3f.getShape(0); ++i)
    {
        for (int j = 0; j < Mat4x5x3f.getShape(1); ++j)
        {
            for (int k = 0; k < Mat4x5x3f.getShape(2); ++k)
            {
                std::cout << std::setw(3) << std::setfill('0') <<  data[ i * 3 * 5 + j * 3 + k] << " ";
                BOOST_REQUIRE_EQUAL(data[i * 3 * 5 + j * 3 + k], 100 * i + 10 * j + k);
            }
        }
        std::cout << std::endl;
    }
    }
}

BOOST_AUTO_TEST_CASE(dynamic_shape_tensor)
{
    {
        ScopedAllocator<TestAllocator> allocator;
    
    tcv::Tensor_<float, -1,-1,-1> Mat4x5x3f(4,5,3);
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getStrideBytes(), 5 * 3 * sizeof(float));
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getStrideBytes(1), 3 * sizeof(float));
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getStrideBytes(2), sizeof(float));
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getNumBytes(0), 4 * 5 * 3 * sizeof(float));
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getNumBytes(1), 3 * 5 * sizeof(float));
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getNumBytes(2), 3 * sizeof(float));
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getShape(0), 4);
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getShape(1), 5);
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getShape(2), 3);
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getShapeBytes(0), 4 * sizeof(float));
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getShapeBytes(1), 5 * sizeof(float));
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getShapeBytes(2), 3 * sizeof(float));
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getNumElements(0), 4 * 5 * 3);
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getNumElements(1), 5 * 3);
    BOOST_REQUIRE_EQUAL(Mat4x5x3f.getNumElements(2), 3);

    for (int i = 0; i < Mat4x5x3f.getShape(0); ++i)
    {
        for (int j = 0; j < Mat4x5x3f.getShape(1); ++j)
        {
            for (int k = 0; k < Mat4x5x3f.getShape(2); ++k)
            {
                BOOST_REQUIRE_EQUAL(Mat4x5x3f.getOffset(i, j, k), k + j * 3 + i * 5 * 3);
            }
        }
    }

    std::cout << "================ Dynamic Shape ===================\n";
    int num_elems = Mat4x5x3f.getNumElements(0);
    const float* data = Mat4x5x3f.getCpu();
    BOOST_REQUIRE(data);
    for (int i = 0; i < Mat4x5x3f.getShape(0); ++i)
    {
        for (int j = 0; j < Mat4x5x3f.getShape(1); ++j)
        {
            for (int k = 0; k < Mat4x5x3f.getShape(2); ++k)
            {
                Mat4x5x3f(i, j, k) = 100 * i + 10 * j + k;
            }
        }
    }
    std::cout << "\noperator() access\n";
    for (int i = 0; i < Mat4x5x3f.getShape(0); ++i)
    {
        for (int j = 0; j < Mat4x5x3f.getShape(1); ++j)
        {
            for (int k = 0; k < Mat4x5x3f.getShape(2); ++k)
            {
                std::cout << std::setw(3) << std::setfill('0') << Mat4x5x3f(i, j, k) << " ";
            }
        }
        std::cout << std::endl;
    }
    std::cout << "\nraw memory access\n";

    for (int i = 0; i < Mat4x5x3f.getShape(0); ++i)
    {
        for (int j = 0; j < Mat4x5x3f.getShape(1); ++j)
        {
            for (int k = 0; k < Mat4x5x3f.getShape(2); ++k)
            {
                std::cout << std::setw(3) << std::setfill('0') << data[i * 3 * 5 + j * 3 + k] << " ";
                BOOST_REQUIRE_EQUAL(data[i * 3 * 5 + j * 3 + k], 100 * i + 10 * j + k);
            }
        }
        std::cout << std::endl;
    }
    }
}

#define DETECT_CONSTEXPR(NAME, FUNC) \
template<typename Trait> \
struct Detect##NAME \
{ \
    template<size_t Value = Trait::FUNC> \
    constexpr static std::true_type do_call(int) { return std::true_type(); } \
    constexpr static std::false_type do_call(...) { return std::false_type(); } \
    static const bool value = do_call(0); \
}

DETECT_CONSTEXPR(StaticDims, getNumDims());

template<class Trait, int N>
struct DetectStaticShape
{
    template<size_t Value = Trait::getStaticShape(N)>
    constexpr static bool do_call(int) { return Trait::getStaticShape(N) != 0; }
    constexpr static bool do_call(...) { return false; }
    static const bool value = do_call(0);
};

BOOST_AUTO_TEST_CASE(static_dim_trait_test)
{
    

}

BOOST_AUTO_TEST_CASE(static_channel_dim)
{
    {
        ScopedAllocator<TestAllocator> allocator;
        tcv::Tensor_<float, -1, -1, 3> Matx3f(340, 480);
        BOOST_REQUIRE_EQUAL(Matx3f.getShape(0), 340);
        BOOST_REQUIRE_EQUAL(Matx3f.getShape(1), 480);
        BOOST_REQUIRE_EQUAL(Matx3f.getShape(2), 3);
    
        BOOST_REQUIRE_EQUAL(Matx3f.getShapeBytes(0), 340 * sizeof(float));
        BOOST_REQUIRE_EQUAL(Matx3f.getShapeBytes(1), 480 * sizeof(float));
        BOOST_REQUIRE_EQUAL(Matx3f.getShapeBytes(2), 3 * sizeof(float));
    
        BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(0), 340 * 480 * 3);
        BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(1), 480 * 3);
        BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(2), 3);
    
        BOOST_REQUIRE_EQUAL(Matx3f.getStride(0), 480 * 3);
        BOOST_REQUIRE_EQUAL(Matx3f.getStride(1), 3);
        BOOST_REQUIRE_EQUAL(Matx3f.getStride(2), 1);

        BOOST_REQUIRE_EQUAL(Matx3f.getStrideBytes(0), 480 * 3 * sizeof(float));
        BOOST_REQUIRE_EQUAL(Matx3f.getStrideBytes(1), 3 * sizeof(float));
        BOOST_REQUIRE_EQUAL(Matx3f.getStrideBytes(2), sizeof(float));


        bool const_shape0 = DetectStaticShape<tcv::StaticShape<float, -1, -1, 3>, 0>::value;
        bool const_shape1 = DetectStaticShape<tcv::StaticShape<float, -1, -1, 3>, 1>::value;
        bool const_shape2 = DetectStaticShape<tcv::StaticShape<float, -1, -1, 3>, 2>::value;
        bool const_shape3 = DetectStaticShape<tcv::StaticShape<float, -1, -1, 3>, 3>::value;
        tcv::StaticShape<float, -1, -1, 3>::getStaticShape(2);
        BOOST_REQUIRE_EQUAL(const_shape0, false);
        BOOST_REQUIRE_EQUAL(const_shape1, false);
        BOOST_REQUIRE_EQUAL(const_shape2, true);
        Matx3f.setShape(500, 400);
        BOOST_REQUIRE_EQUAL(Matx3f.getShape(0), 500);
        BOOST_REQUIRE_EQUAL(Matx3f.getShape(1), 400);
        BOOST_REQUIRE_EQUAL(Matx3f.getShape(2), 3);
        BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(0), 500 * 400 * 3);
        BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(1), 400 * 3);
        BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(2), 3);
    }    
}

BOOST_AUTO_TEST_CASE(tensor_downconvert)
{
    {
        ScopedAllocator<TestAllocator> allocator;

        tcv::Tensor mat;
        {
            tcv::Tensor_<float, -1, -1, 3> Matx3f(340, 480);

            BOOST_REQUIRE_EQUAL(Matx3f.getShape(0), 340);
            BOOST_REQUIRE_EQUAL(Matx3f.getShape(1), 480);
            BOOST_REQUIRE_EQUAL(Matx3f.getShape(2), 3);

            BOOST_REQUIRE_EQUAL(Matx3f.getShapeBytes(0), 340 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f.getShapeBytes(1), 480 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f.getShapeBytes(2), 3 * sizeof(float));

            BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(0), 340 * 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(1), 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(2), 3);

            BOOST_REQUIRE_EQUAL(Matx3f.getStride(0), 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getStride(1), 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getStride(2), 1);

            BOOST_REQUIRE_EQUAL(Matx3f.getStrideBytes(0), 480 * 3 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f.getStrideBytes(1), 3 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f.getStrideBytes(2), sizeof(float));

            BOOST_REQUIRE_EQUAL(Matx3f.getSize(0), 340 * 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getSize(1), 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getSize(2), 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getNumDims(), 3);

            tcv::Tensor_<float, -1, -1, 3> Matx3f2(Matx3f);

            BOOST_REQUIRE_EQUAL(Matx3f2.getShape(0), 340);
            BOOST_REQUIRE_EQUAL(Matx3f2.getShape(1), 480);
            BOOST_REQUIRE_EQUAL(Matx3f2.getShape(2), 3);

            BOOST_REQUIRE_EQUAL(Matx3f2.getShapeBytes(0), 340 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f2.getShapeBytes(1), 480 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f2.getShapeBytes(2), 3 * sizeof(float));

            BOOST_REQUIRE_EQUAL(Matx3f2.getNumElements(0), 340 * 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f2.getNumElements(1), 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f2.getNumElements(2), 3);

            BOOST_REQUIRE_EQUAL(Matx3f2.getStride(0), 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f2.getStride(1), 3);
            BOOST_REQUIRE_EQUAL(Matx3f2.getStride(2), 1);

            BOOST_REQUIRE_EQUAL(Matx3f2.getStrideBytes(0), 480 * 3 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f2.getStrideBytes(1), 3 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f2.getStrideBytes(2), sizeof(float));

            BOOST_REQUIRE_EQUAL(Matx3f2.getSize(0), 340 * 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f2.getSize(1), 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f2.getSize(2), 3);
            BOOST_REQUIRE_EQUAL(Matx3f2.getNumDims(), 3);

            tcv::Tensor base(Matx3f);
            for(int i = 0; i < Matx3f.getShape(0); ++i)
            {
                for(int j = 0;  j < Matx3f.getShape(1); ++j)
                {
                    for(int k = 0; k < Matx3f.getShape(2); ++k)
                    {
                        Matx3f(i,j,k) = 100*i + 10 * j + k;
                    }
                }
            }
            BOOST_REQUIRE_EQUAL(Matx3f.getShape(0), 340);
            BOOST_REQUIRE_EQUAL(Matx3f.getShape(1), 480);
            BOOST_REQUIRE_EQUAL(Matx3f.getShape(2), 3);

            BOOST_REQUIRE_EQUAL(Matx3f.getShapeBytes(0), 340 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f.getShapeBytes(1), 480 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f.getShapeBytes(2), 3 * sizeof(float));

            BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(0), 340 * 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(1), 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(2), 3);

            BOOST_REQUIRE_EQUAL(Matx3f.getStride(0), 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getStride(1), 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getStride(2), 1);

            BOOST_REQUIRE_EQUAL(Matx3f.getStrideBytes(0), 480 * 3 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f.getStrideBytes(1), 3 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f.getStrideBytes(2), sizeof(float));

            BOOST_REQUIRE_EQUAL(base.getShape(0), 340);
            BOOST_REQUIRE_EQUAL(base.getShape(1), 480);
            BOOST_REQUIRE_EQUAL(base.getShape(2), 3);

            BOOST_REQUIRE_EQUAL(base.getShapeBytes(0), 340 * sizeof(float));
            BOOST_REQUIRE_EQUAL(base.getShapeBytes(1), 480 * sizeof(float));
            BOOST_REQUIRE_EQUAL(base.getShapeBytes(2), 3 * sizeof(float));

            BOOST_REQUIRE_EQUAL(base.getNumElements(0), 340 * 480 * 3);
            BOOST_REQUIRE_EQUAL(base.getNumElements(1), 480 * 3);
            BOOST_REQUIRE_EQUAL(base.getNumElements(2), 3);

            BOOST_REQUIRE_EQUAL(base.getStride(0), 480 * 3);
            BOOST_REQUIRE_EQUAL(base.getStride(1), 3);
            BOOST_REQUIRE_EQUAL(base.getStride(2), 1);

            BOOST_REQUIRE_EQUAL(base.getStrideBytes(0), 480 * 3 * sizeof(float));
            BOOST_REQUIRE_EQUAL(base.getStrideBytes(1), 3 * sizeof(float));
            BOOST_REQUIRE_EQUAL(base.getStrideBytes(2), sizeof(float));

            BOOST_REQUIRE_EQUAL(Matx3f.getNumBytes(0), base.getNumBytes(0));
            BOOST_REQUIRE_EQUAL(Matx3f.getNumBytes(1), base.getNumBytes(1));
            BOOST_REQUIRE_EQUAL(Matx3f.getNumBytes(2), base.getNumBytes(2));

            BOOST_REQUIRE_EQUAL(Matx3f.getNumDims(), base.getNumDims());

            mat = Matx3f;
            // Makesure matx3f hasn't changed
            BOOST_REQUIRE_EQUAL(Matx3f.getShape(0), 340);
            BOOST_REQUIRE_EQUAL(Matx3f.getShape(1), 480);
            BOOST_REQUIRE_EQUAL(Matx3f.getShape(2), 3);

            BOOST_REQUIRE_EQUAL(Matx3f.getShapeBytes(0), 340 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f.getShapeBytes(1), 480 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f.getShapeBytes(2), 3 * sizeof(float));

            BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(0), 340 * 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(1), 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(2), 3);

            BOOST_REQUIRE_EQUAL(Matx3f.getStride(0), 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getStride(1), 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getStride(2), 1);

            BOOST_REQUIRE_EQUAL(Matx3f.getStrideBytes(0), 480 * 3 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f.getStrideBytes(1), 3 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f.getStrideBytes(2), sizeof(float));

            BOOST_REQUIRE_EQUAL(Matx3f.getSize(0), 340 * 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getSize(1), 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getSize(2), 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getNumDims(), 3);

            BOOST_REQUIRE_EQUAL(Matx3f.getNumBytes(0), mat.getNumBytes(0));
            BOOST_REQUIRE_EQUAL(Matx3f.getNumBytes(1), mat.getNumBytes(1));
            BOOST_REQUIRE_EQUAL(Matx3f.getNumBytes(2), mat.getNumBytes(2));

            BOOST_REQUIRE_EQUAL(Matx3f.getNumDims(), mat.getNumDims());

            // Make sure calculated values are the same between mats
            BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(0), mat.getNumElements(0));
            BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(1), mat.getNumElements(1));
            BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(2), mat.getNumElements(2));

            BOOST_REQUIRE_EQUAL(Matx3f.getShape(0), mat.getShape(0));
            BOOST_REQUIRE_EQUAL(Matx3f.getShape(1), mat.getShape(1));
            BOOST_REQUIRE_EQUAL(Matx3f.getShape(2), mat.getShape(2));

            BOOST_REQUIRE_EQUAL(Matx3f.getShapeBytes(0), mat.getShapeBytes(0));
            BOOST_REQUIRE_EQUAL(Matx3f.getShapeBytes(1), mat.getShapeBytes(1));
            BOOST_REQUIRE_EQUAL(Matx3f.getShapeBytes(2), mat.getShapeBytes(2));

            BOOST_REQUIRE_EQUAL(Matx3f.getStride(0), mat.getStride(0));
            BOOST_REQUIRE_EQUAL(Matx3f.getStride(1), mat.getStride(1));
            BOOST_REQUIRE_EQUAL(Matx3f.getStride(2), mat.getStride(2));

            BOOST_REQUIRE_EQUAL(Matx3f.getStrideBytes(0), mat.getStrideBytes(0));
            BOOST_REQUIRE_EQUAL(Matx3f.getStrideBytes(1), mat.getStrideBytes(1));
            BOOST_REQUIRE_EQUAL(Matx3f.getStrideBytes(2), mat.getStrideBytes(2));
            BOOST_REQUIRE_EQUAL(Matx3f.getCpu(0), (const float*)mat.getCpu(0));

            BOOST_REQUIRE_EQUAL(mat.elemType(), tcv::DataType<float>::DType);

            BOOST_REQUIRE_EQUAL(mat.getShape(0), 340);
            BOOST_REQUIRE_EQUAL(mat.getShape(1), 480);
            BOOST_REQUIRE_EQUAL(mat.getShape(2), 3);

            BOOST_REQUIRE_EQUAL(mat.getShapeBytes(0), 340 * sizeof(float));
            BOOST_REQUIRE_EQUAL(mat.getShapeBytes(1), 480 * sizeof(float));
            BOOST_REQUIRE_EQUAL(mat.getShapeBytes(2), 3 * sizeof(float));

            BOOST_REQUIRE_EQUAL(mat.getNumElements(0), 340 * 480 * 3);
            BOOST_REQUIRE_EQUAL(mat.getNumElements(1), 480 * 3);
            BOOST_REQUIRE_EQUAL(mat.getNumElements(2), 3);

            BOOST_REQUIRE_EQUAL(mat.getStride(0), 480 * 3);
            BOOST_REQUIRE_EQUAL(mat.getStride(1), 3);
            BOOST_REQUIRE_EQUAL(mat.getStride(2), 1);

            BOOST_REQUIRE_EQUAL(mat.getStrideBytes(0), 480 * 3 * sizeof(float));
            BOOST_REQUIRE_EQUAL(mat.getStrideBytes(1), 3 * sizeof(float));
            BOOST_REQUIRE_EQUAL(mat.getStrideBytes(2), sizeof(float));

            BOOST_REQUIRE_EQUAL(mat.getSize(0), 340 * 480 * 3);
            BOOST_REQUIRE_EQUAL(mat.getSize(1), 480 * 3);
            BOOST_REQUIRE_EQUAL(mat.getSize(2), 3);
            BOOST_REQUIRE_EQUAL(mat.getNumDims(), 3);

            BOOST_REQUIRE_EQUAL(mat.getShape(0), 340);
            BOOST_REQUIRE_EQUAL(mat.getShape(1), 480);
            BOOST_REQUIRE_EQUAL(mat.getShape(2), 3);


        }
        const float* data = (const float*)mat.getCpu(0);
        int step0 = mat.getStride(0);
        int step1 = mat.getStride(1);
        for (int i = 0; i < mat.getShape(0); ++i)
        {
            for (int j = 0; j < mat.getShape(1); ++j)
            {
                for (int k = 0; k < mat.getShape(2); ++k)
                {
                    BOOST_REQUIRE_EQUAL(data[i * step0 + j * step1 + k], 100 * i + 10 * j + k);
                }
            }
        }
        {
            tcv::Tensor_<float, -1, -1, 3> Matx3f(mat);
            // Make sure mat is unchanged
            BOOST_REQUIRE_EQUAL(mat.getShape(0), 340);
            BOOST_REQUIRE_EQUAL(mat.getShape(1), 480);
            BOOST_REQUIRE_EQUAL(mat.getShape(2), 3);

            BOOST_REQUIRE_EQUAL(mat.getShapeBytes(0), 340 * sizeof(float));
            BOOST_REQUIRE_EQUAL(mat.getShapeBytes(1), 480 * sizeof(float));
            BOOST_REQUIRE_EQUAL(mat.getShapeBytes(2), 3 * sizeof(float));

            BOOST_REQUIRE_EQUAL(mat.getNumElements(0), 340 * 480 * 3);
            BOOST_REQUIRE_EQUAL(mat.getNumElements(1), 480 * 3);
            BOOST_REQUIRE_EQUAL(mat.getNumElements(2), 3);

            BOOST_REQUIRE_EQUAL(mat.getStride(0), 480 * 3);
            BOOST_REQUIRE_EQUAL(mat.getStride(1), 3);
            BOOST_REQUIRE_EQUAL(mat.getStride(2), 1);

            BOOST_REQUIRE_EQUAL(mat.getStrideBytes(0), 480 * 3 * sizeof(float));
            BOOST_REQUIRE_EQUAL(mat.getStrideBytes(1), 3 * sizeof(float));
            BOOST_REQUIRE_EQUAL(mat.getStrideBytes(2), sizeof(float));

            BOOST_REQUIRE_EQUAL(mat.getSize(0), 340 * 480 * 3);
            BOOST_REQUIRE_EQUAL(mat.getSize(1), 480 * 3);
            BOOST_REQUIRE_EQUAL(mat.getSize(2), 3);
            BOOST_REQUIRE_EQUAL(mat.getNumDims(), 3);

            BOOST_REQUIRE_EQUAL(mat.getShape(0), 340);
            BOOST_REQUIRE_EQUAL(mat.getShape(1), 480);
            BOOST_REQUIRE_EQUAL(mat.getShape(2), 3);

            // makesure Matx3f is valid and correct
            BOOST_REQUIRE_EQUAL(Matx3f.getShapeBytes(0), 340 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f.getShapeBytes(1), 480 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f.getShapeBytes(2), 3 * sizeof(float));

            BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(0), 340 * 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(1), 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(2), 3);

            BOOST_REQUIRE_EQUAL(Matx3f.getStride(0), 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getStride(1), 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getStride(2), 1);

            BOOST_REQUIRE_EQUAL(Matx3f.getStrideBytes(0), 480 * 3 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f.getStrideBytes(1), 3 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f.getStrideBytes(2), sizeof(float));

            BOOST_REQUIRE_EQUAL(Matx3f.getSize(0), 340 * 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getSize(1), 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getSize(2), 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getNumDims(), 3);

            BOOST_REQUIRE_EQUAL(Matx3f.getCpu(0), (const float*)mat.getCpu(0));
            const float* data = Matx3f.getCpu(0);
            int step0 = Matx3f.getStride(0);
            int step1 = Matx3f.getStride(1);
            for (int i = 0; i < Matx3f.getShape(0); ++i)
            {
                for (int j = 0; j < Matx3f.getShape(1); ++j)
                {
                    for (int k = 0; k < Matx3f.getShape(2); ++k)
                    {
                        BOOST_REQUIRE_EQUAL(data[i * step0 + j * step1 + k], 100 * i + 10 * j + k);
                        BOOST_REQUIRE_EQUAL(Matx3f(i,j,k), 100 * i + 10 * j + k);
                    }
                }
            }
        }
        {
            tcv::Tensor_<float3, -1, -1> Matx3f(340, 480);
            BOOST_REQUIRE_EQUAL(Matx3f.getShape(0), 340);
            BOOST_REQUIRE_EQUAL(Matx3f.getShape(1), 480);
            BOOST_REQUIRE_EQUAL(Matx3f.getShape(2), 3);

            BOOST_REQUIRE_EQUAL(Matx3f.getShapeBytes(0), 340 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f.getShapeBytes(1), 480 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f.getShapeBytes(2), 3 * sizeof(float));

            BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(0), 340 * 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(1), 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getNumElements(2), 3);

            BOOST_REQUIRE_EQUAL(Matx3f.getStride(0), 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getStride(1), 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getStride(2), 1);

            BOOST_REQUIRE_EQUAL(Matx3f.getStrideBytes(0), 480 * 3 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f.getStrideBytes(1), 3 * sizeof(float));
            BOOST_REQUIRE_EQUAL(Matx3f.getStrideBytes(2), sizeof(float));

            BOOST_REQUIRE_EQUAL(Matx3f.getSize(0), 340 * 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getSize(1), 480 * 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getSize(2), 3);
            BOOST_REQUIRE_EQUAL(Matx3f.getNumDims(), 3);

            tcv::Tensor_<TestType, -1, -1> VecMat(Matx3f);
            BOOST_REQUIRE_EQUAL(VecMat.getShape(0), 340);
            BOOST_REQUIRE_EQUAL(VecMat.getShape(1), 480);
            BOOST_REQUIRE_EQUAL(VecMat.getShape(2), 3);

            BOOST_REQUIRE_EQUAL(VecMat.getShapeBytes(0), 340 * sizeof(float));
            BOOST_REQUIRE_EQUAL(VecMat.getShapeBytes(1), 480 * sizeof(float));
            BOOST_REQUIRE_EQUAL(VecMat.getShapeBytes(2), 3 * sizeof(float));

            BOOST_REQUIRE_EQUAL(VecMat.getNumElements(0), 340 * 480 * 3);
            BOOST_REQUIRE_EQUAL(VecMat.getNumElements(1), 480 * 3);
            BOOST_REQUIRE_EQUAL(VecMat.getNumElements(2), 3);

            BOOST_REQUIRE_EQUAL(VecMat.getStride(0), 480 * 3);
            BOOST_REQUIRE_EQUAL(VecMat.getStride(1), 3);
            BOOST_REQUIRE_EQUAL(VecMat.getStride(2), 1);

            BOOST_REQUIRE_EQUAL(VecMat.getStrideBytes(0), 480 * 3 * sizeof(float));
            BOOST_REQUIRE_EQUAL(VecMat.getStrideBytes(1), 3 * sizeof(float));
            BOOST_REQUIRE_EQUAL(VecMat.getStrideBytes(2), sizeof(float));

            BOOST_REQUIRE_EQUAL(VecMat.getSize(0), 340 * 480 * 3);
            BOOST_REQUIRE_EQUAL(VecMat.getSize(1), 480 * 3);
            BOOST_REQUIRE_EQUAL(VecMat.getSize(2), 3);
            BOOST_REQUIRE_EQUAL(VecMat.getNumDims(), 3);

            //VecMat.setShape(500,500);
        
        }
    }
}
#ifdef WITH_OPENCV
template<class T, int N>
std::ostream& operator<<(std::ostream& os, const cv::Vec<T, N>& elem)
{
    os << "(";
    for(int i = 0; i < N; ++i)
    {
        if(i != 0)
            os << ", ";
        os << elem[i];
    }
    os << ")";
    return os;
}
#else


std::ostream& operator<<(std::ostream& os, const float3& elem)
{
    os << "(" << elem.x << ", " << elem.y << ", " << elem.z << ")";
    return os;
}


#endif
BOOST_AUTO_TEST_CASE(vector)
{
    {
        ScopedAllocator<TestAllocator> allocator;
        {
            tcv::Tensor_<float, 100> vec;

            BOOST_REQUIRE_EQUAL(vec.getShape(0), 100);
            BOOST_REQUIRE_EQUAL(vec.getShape(1), 1);
            BOOST_REQUIRE_EQUAL(vec.getShapeBytes(0), 100 * sizeof(float));
            BOOST_REQUIRE_EQUAL(vec.getShapeBytes(1), sizeof(float));
            BOOST_REQUIRE_EQUAL(vec.getNumElements(0), 100);
            BOOST_REQUIRE_EQUAL(vec.getNumElements(1), 1);
            BOOST_REQUIRE_EQUAL(vec.getStride(0), 1);
            BOOST_REQUIRE_EQUAL(vec.getStride(1), 1);
            BOOST_REQUIRE_EQUAL(vec.getStrideBytes(0), sizeof(float));
            BOOST_REQUIRE_EQUAL(vec.getStrideBytes(1), sizeof(float));
            BOOST_REQUIRE_EQUAL(vec.getSize(0), 100);
            BOOST_REQUIRE_EQUAL(vec.getSize(1), 1);
            BOOST_REQUIRE_EQUAL(vec.getNumDims(), 1);
        }
    
        {

            tcv::Tensor_<TestType, 100> vec;

            BOOST_REQUIRE_EQUAL(vec.getShape(0), 100);
            BOOST_REQUIRE_EQUAL(vec.getShape(1), 3);
            BOOST_REQUIRE_EQUAL(vec.getShapeBytes(0), 100  * sizeof(float));
            BOOST_REQUIRE_EQUAL(vec.getShapeBytes(1), 3 * sizeof(float));
            BOOST_REQUIRE_EQUAL(vec.getNumElements(0), 100 * 3);
            BOOST_REQUIRE_EQUAL(vec.getNumElements(1), 3);
            BOOST_REQUIRE_EQUAL(vec.getStride(0), 3);
            BOOST_REQUIRE_EQUAL(vec.getStride(1), 1);
            BOOST_REQUIRE_EQUAL(vec.getStrideBytes(0), 3 * sizeof(float));
            BOOST_REQUIRE_EQUAL(vec.getStrideBytes(1), sizeof(float));
            BOOST_REQUIRE_EQUAL(vec.getSize(0), 100 * 3);
            BOOST_REQUIRE_EQUAL(vec.getSize(1), 3);

            BOOST_REQUIRE_EQUAL(vec.getNumDims(), 2);
            for(int i = 0; i < 100; ++i)
            {
                TestType& elem = vec(i);
#ifdef WITH_OPENCV
                elem[0] = i * 3;
                elem[1] = i * 3 + 1;
                elem[2] = i * 3 + 2;
#else
                elem.x = i * 3;
                elem.y = i * 3 + 1;
                elem.z = i * 3 + 2;
#endif
            }
            for(int i = 0; i < 100; ++i)
            {
                std::cout << vec(i) << " ";
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(check_memory_leak)
{
    delete Allocator::getDefaultAllocator();
}
