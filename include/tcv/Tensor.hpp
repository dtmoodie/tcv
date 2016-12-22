#pragma once
#include "SyncedMemory.hpp"
#include "TypeTraits.hpp"
#include "Defs.hpp"
#include "Allocator.hpp"

#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <cassert>

namespace tcv
{

    template<typename T>
    void Offset(size_t& offset, int index, const size_t* stride, T t)
    {
        offset += stride[index] * t;
    }
    template<typename T, typename ...ArgTypes>
    void Offset(size_t& offset, int index, const size_t* stride, T t, ArgTypes... args)
    {
        offset += stride[index] * t;
        Offset(offset, index + 1, stride, args...);
    }

    template<int Dim, int MaxDim, typename T>
    size_t Offset_(const size_t* stride, T idx)
    {
        static_assert(Dim < MaxDim, "Dim must be less than MaxDim");
        return stride[Dim] * idx;
    }
    template<int Dim, int MaxDim, typename T, typename... ArgTypes>
    size_t Offset_(const size_t* stride, T idx, ArgTypes... args)
    {
        return Offset_<Dim + 1, MaxDim, ArgTypes...>(stride, args...) + stride[Dim] * idx;
    }
    template<int MaxDim, typename... ArgTypes>
    size_t Offset(const size_t* stride, ArgTypes... args)
    {
        return Offset_<0, MaxDim, ArgTypes...>(stride, args...);
    }


    class SyncedMemory;
    class Tensor;
    
    class Tensor
    {
    public:
        Tensor();
        ~Tensor();
        size_t numBytes() const
        {
            size_t size = 1;
            for(int i = 0; i < dims; ++i)
            {
                size *= (*shape)[i];
            }
            return size;
        }
        uint8_t elemType() const
        {
            return data->getElemType();
        }
        
        template<typename T> T& at(const std::initializer_list<int>& idx)
        {
            size_t offset = 0;
            int i = 0;
            for(auto itr = idx.begin(); itr != idx.end(); ++itr, ++i)
            {
                offset += *itr * (*stride)[i];
            }
            return data->getCpu() + offset;
        }

        template<typename T, int N> T& at(int indecies[N])
        {
            size_t offset = 0;
            for(int i = N-1; i >= 0; --i)
            {
                offset += indecies[i] * (*stride)[i];
            }
            return *(static_cast<T*>(data->getCpu() + offset));
        }

        SyncedMemory* data;
        SyncedMemory_<size_t>* stride;
        SyncedMemory_<size_t>* shape;
        int dims;
        Allocator* allocator;
        uint64_t flags;
        int* refCount;
        void cleanup();
    };

    template<class T, int N>
    struct Shape
    {
        size_t _shape[N];
        size_t _stride[N];
    };
    
    template<class T>
    struct Shape<T, -1>
    {
        
    };

    template<class T, int N, int... Dims> struct DimHandle: public DimHandle<T, Dims...>
    {
        template<class Arg, class...Args>
        static constexpr size_t offset(Arg arg, Args... args)
        {
            return DimHandle<T,Dims...>::offset(args...) + arg*stride();
        }
        static constexpr size_t stride()
        {
            return DimHandle<T, Dims...>::stride()*N;
        }
    };
    template<class T, int N> struct DimHandle<T, N>
    {
        template<class Arg>
        static constexpr size_t offset(Arg arg)
        {
            return arg*N;
        }
        static constexpr size_t stride()
        {
            return sizeof(T)*N;
        }
    };

    template<class T, int... Size>
    struct StaticShape
    {
        static const int N = sizeof...(Size);
        static const int _flags = STATIC_SHAPE;
        StaticShape()
        {
            size_t size = sizeof(T);
            for(int i = N - 1; i >= 0; --i)
            {
                _stride[i] = size;
                size *= _shape[i];
            }
        }
        template<class...Args>
        constexpr size_t getOffset(Args... args)
        {
            return DimHandle<T, Size...>::offset(args...);
        }

        constexpr size_t getStride()
        {
            return DimHandle<T, Size...>::stride();
        }

        size_t _shape[N] = {Size...};
        size_t _stride[N];
    };

    
    template<class T, class Shape> class Tensor_
        : public Tensor
        , public Shape
    {
    public:
        Tensor_():
            Shape()
        {
            this->shape = new SyncedMemory_<size_t>();
            this->stride = new SyncedMemory_<size_t>();
            this->shape->setCpuData(this->_shape, Shape::N);
            this->stride->setCpuData(this->_stride, Shape::N);
            this->dims = Shape::N;
            this->flags = Shape::_flags;
            this->flags += DataType<T>::DType;
            if(Shape::_flags & STATIC_SHAPE)
            {
                Allocator::getDefaultAllocator()->allocate(this, numBytes(), DataType<T>::DType);
            }
        }

        template<typename ... ArgTypes>
        T& operator()(ArgTypes... args)
        {
            size_t offset = Offset<Shape::N>(this->_stride, args...);
            size_t test = Shape::getOffset(args...);
            return *(T*)(data->getCpu() + offset);
        }

        template<typename ... ArgTypes>
        const T& operator()(ArgTypes... args) const
        {
            size_t offset = Offset<Shape::N>(this->_stride, args...);

            return *(T*)(data->getCpu() + offset);
        }
    };
}
