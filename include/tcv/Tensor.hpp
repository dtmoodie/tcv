#pragma once
#include "SyncedMemory.hpp"
#include "TypeTraits.hpp"
#include "Defs.hpp"
#include "Allocator.hpp"
#include "Shape.hpp"
#include "yar/Logging.hpp"
#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <cassert>

namespace tcv
{
    template<int Dim, int MaxDim, typename T>
    size_t offset_(const size_t* stride, T idx)
    {
        static_assert(Dim < MaxDim, "Dim must be less than MaxDim");
        return stride[Dim] * idx;
    }
    template<int Dim, int MaxDim, typename T, typename... ArgTypes>
    size_t offset_(const size_t* stride, T idx, ArgTypes... args)
    {
        return offset_<Dim + 1, MaxDim, ArgTypes...>(stride, args...) + stride[Dim] * idx;
    }
    template<int MaxDim, typename... ArgTypes>
    size_t offset(const size_t* stride, ArgTypes... args)
    {
        return offset_<0, MaxDim, ArgTypes...>(stride, args...);
    }

    inline size_t offset(const size_t* stride, int dims, size_t idx)
    {
        return *stride * idx;
    }

    template<typename... ArgTypes>
    size_t offset(const size_t* stride, int dims, size_t idx, ArgTypes... args)
    {
        return (*stride * idx) + offset(stride + 1, dims - 1, args...);
    }
    inline size_t getSize(const size_t* stride, int dims)
    {
        size_t output = 1;
        for(int i = 0; i < dims; ++i)
        {
            output *= stride[i];
        }
        return output;
    }

    class SyncedMemory;
    class Tensor;
    template<class T, int64_t... Sizes> class Tensor_;

    class Tensor
    {
    public:
        
        template<class T, int64_t... Sizes>
        Tensor(Tensor_<T, Sizes...>& tensor):
            Tensor(static_cast<Tensor&>(tensor))
        {
            flags |= DataType<T>::DType << 12;
        }
        Tensor(Tensor& tensor);
        Tensor();
        ~Tensor();
        size_t numBytes() const;
        uint8_t elemType() const;
        
        const uint8_t* getCpu(cudaStream_t stream) const;
        uint8_t* getCpuMutable(cudaStream_t stream);

        const uint8_t* getGpu(cudaStream_t stream) const;
        uint8_t* getGpuMutable(cudaStream_t stream);

        size_t getStride(int dim = 0);
        size_t getStrideBytes(int dim = 0);

        size_t getNumBytes(int dim = 0);

        size_t getNumElements(int dim = 0);
        size_t getShape(int dim = 0);
        static size_t getStaticShape(int dim = 0);
        size_t getShapeBytes(int dim = 0);
        size_t getNumDims();
        size_t getSize(int dim = 0);

        template<class T, int64_t... Sizes>
        Tensor& operator=(Tensor_<T, Sizes...>& tensor)
        {
            this->operator=(static_cast<Tensor&>(tensor));
            flags |= DataType<T>::DType << 12;
            return *this;
        }
        Tensor& operator=(Tensor& tensor);
        SyncedMemory* data;
        SyncedMemory_<size_t>* stride;
        SyncedMemory_<size_t>* shape;
        int dims;
        Allocator* allocator;
        uint64_t flags;
        int* refCount;
        void cleanup();
        void decrement();
        void resize(size_t size);
    };

    template<class T, int64_t... Sizes> class Tensor_
        : protected Tensor
        , protected StaticShape<T, Sizes...>
    {
    public:
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
        Tensor_():
            Tensor(),
            StaticShape<T, Sizes...>(*shape, *stride)
        {
            this->dims = getNumDims();
            this->flags = shapeFlags();
            flags |= DataType<T>::DType << 12;
            if(flags & STATIC_SHAPE)
            {
                Allocator::getDefaultAllocator()->allocate(this, numBytes(), DataType<T>::DType);
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
            return *(T*)(data->getCpu() + Offset<getNumDims()>(this->_stride, args...));
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
            size_t stride = Shape::getStrideBytes(dim) / sizeof(T);
            return Iterator((T*)data->getCpu(), stride);
        }
        Iterator end(int dim = -1)
        {
            size_t stride = Shape::getStrideBytes(dim) / sizeof(T);
            size_t size = Shape::getNumElements(dim);
            return Iterator((T*)data->getCpu() + size, stride);
        }
    protected:
        friend class Tensor;
        explicit operator Tensor&() const 
        {
            return *this;
        }
    };
}
