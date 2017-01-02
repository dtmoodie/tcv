#pragma once
#include "Tensor.hpp"
#include "Shape.hpp"
#include "Allocator.hpp"
#include "TypeTraits.hpp"
#include "Offsets.hpp"

#include "yar/Logging.hpp"
namespace tcv
{

template<class T, int64_t... Sizes> class Tensor_
    : protected Tensor
    , protected StaticShape<T, Sizes...>
{
public:
    template<class... Args>
    Tensor_(Allocator* allocator, Args... args) :
        Tensor(allocator),
        StaticShape<T, Sizes...>(*shape, *stride, args...)
    {
        this->dims = getNumDims();
        this->flags = shapeFlags();
        flags |= DataType<T>::DType << 12;
        allocator->allocate(this, numBytes(), DataType<T>::DType);
    }
    template<class... Args>
    Tensor_(Args... args):
        Tensor(),
        StaticShape<T, Sizes...>(*shape, *stride, args...)
    {
        this->dims = getNumDims();
        this->flags = shapeFlags();
        flags |= DataType<T>::DType << 12;
        Allocator::getDefaultAllocator()->allocate(this, numBytes(), DataType<T>::DType);
    }
    Tensor_(Allocator* allocator_ = nullptr):
        Tensor(allocator_),
        StaticShape<T, Sizes...>(*shape, *stride)
    {
        this->dims = getNumDims();
        this->flags = shapeFlags();
        flags |= DataType<T>::DType << 12;
        if(flags & STATIC_SHAPE)
        {
            allocator->allocate(this, numBytes(), DataType<T>::DType);
        }
    }
    Tensor_(Tensor& other):
        Tensor(other),
        StaticShape<T, Sizes...>(shape, stride)
    {
        ASSERT_EQ(other.dims, this->getNumDims());
        for(int i = 0; i < other.dims; ++i)
        {
            if(this->getStaticShape(i) != 0)
            {
                ASSERT_EQ(other.getShape(i), this->getStaticShape(i))
                    << " static shape set for dimension " << i << " and shape doesn't match";
            }else
            {
                this->setDimSize(i, other.getShape(i));
            }
        }
    }
    template<class U, int64_t... Size>
    Tensor_(Tensor_<U, Size...>& other):
        Tensor(other),
        StaticShape<T, Sizes...>(shape, stride)
    {
        static_assert(DataType<U>::DType == DataType<T>::DType,
            "Fundamental data types must match for this constructor");
        static_assert(DataType<U>::channels == DataType<T>::channels,
            "Number of channels must match for this constructor");
        for (int i = 0; i < other.getNumDims(); ++i)
        {
            // Check that static dimensions are the same between the two tensors
            if (this->getStaticShape(i) != 0)
            {
                ASSERT_EQ(other.getShape(i), this->getShape(i))
                  << " static shape set for dimension " << i << " and shape doesn't match";
            }
            else
            {
                this->setDimSize(i, other.getShape(i));
            }
        }
    }

    Tensor_(Tensor_<T, Sizes...>& other):
        Tensor(other),
        StaticShape<T, Sizes...>(shape, stride)
    {
        for (int i = 0; i < other.getNumDims(); ++i)
        {
            if (this->getStaticShape(i) != 0)
            {
                //ASSERT_EQ(other.getShape(i), this->getStaticShape(i))
                  //  << " static shape set for dimension " << i << " and shape doesn't match";
                //static_assert(this->getNumDims(i) == other.getNumDims(i));
            }
            else
            {
                this->setDimSize(i, other.getShape(i));
            }
        }
    }

    template<typename ... ArgTypes>
    T& operator()(ArgTypes... args)
    {
        return *((T*)(data->getCpu()) + StaticShape<T, Sizes...>::getOffset(args...));
    }

    template<typename ... ArgTypes>
    const T& operator()(ArgTypes... args) const
    {
        return *(T*)(data->getCpu() + offset<getNumDims()>(this->_stride, args...));
    }
    template<class...Args>
    size_t getOffset(Args... args)
    {
        return tcv::StaticShape<T, Sizes...>::getOffset(args...);
    }
    const T* getCpu(cudaStream_t stream = nullptr) const
    {
        return (const T*)this->Tensor::getCpu(stream);
    }
    T* getCpuMutable(cudaStream_t stream = nullptr)
    {
        return (T*)Tensor::getCpuMutable(stream);
    }
    const T* getGpu(cudaStream_t stream = nullptr) const
    {
        return (const T*)Tensor::getGpu(stream);
    }
    T* getGpuMutable(cudaStream_t stream = nullptr)
    {
        return (T*)Tensor::getGpuMutable(stream);
    }

    size_t getStride(int dim = 0)
    {
        return StaticShape<T, Sizes...>::getStride(dim);
    }
    size_t getStrideBytes(int dim = 0)
    {
        return StaticShape<T, Sizes...>::getStrideBytes(dim);
    }

    size_t getNumBytes(int dim = 0)
    {
        return StaticShape<T, Sizes...>::getNumBytes(dim);
    }

    size_t getNumElements(int dim = 0)
    {
        return StaticShape<T, Sizes...>::getNumElements(dim);
    }
    size_t getShape(int dim = 0)
    {
        return StaticShape<T, Sizes...>::getShape(dim);
    }
    template<class...Args>
    void setShape(Args... args)
    {
        static_assert(sizeof...(Args) <= sizeof...(Sizes),
            "Cannot increase dimensions of this type of tensor");
        StaticShape<T, Sizes...>::setShape(args...);
        Tensor::resize(Tensor::getNumBytes());
    }
    size_t getSize(int dim)
    {
        return StaticShape<T, Sizes...>::getSize(dim);
    }
    static size_t getStaticShape(int dim = 0)
    {
        return StaticShape<T, Sizes...>::getStaticShape(dim);
    }
    size_t getShapeBytes(int dim = 0)
    {
        return StaticShape<T, Sizes...>::getShapeBytes(dim);
    }
    constexpr static size_t getNumDims()
    {
        return StaticShape<T, Sizes...>::getNumDims();
    }
    constexpr static int shapeFlags()
    {
        return StaticShape<T, Sizes...>::shapeFlags();
    }

    // Massages the data into a new given size.  Will throw an exception if an incompatible size is
    // passed in
    template<class... Args>
    void reshape(Args... args)
    {
        //setShape(args...);
    }
    struct Iterator
    {
        Iterator(T* ptr, size_t stride_) :
            begin(ptr), stride(stride_) {}
        Iterator& operator++()
        {
            begin += stride;
        }
        T& operator*()
        {
            return *begin;
        }
    protected:

        T* begin;
        size_t stride;
    };
    Iterator begin(int dim = -1)
    {
        size_t stride = StaticShape<T, Sizes...>::getStrideBytes(dim) / sizeof(T);
        return Iterator((T*)data->getCpu(), stride);
    }
    Iterator end(int dim = -1)
    {
        size_t stride = StaticShape<T, Sizes...>::getStrideBytes(dim) / sizeof(T);
        size_t size = StaticShape<T, Sizes...>::getNumElements(dim);
        return Iterator((T*)data->getCpu() + size, stride);
    }
protected:
    friend class Tensor;
    explicit operator Tensor&() const
    {
        return *this;
    }
};
} // namespace tcv
