#pragma once
#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <cassert>
#include "SyncedMemory.hpp"



namespace tcv
{
    enum DTypes : short
    {
        U8  = 0,
        S8  = 1,
        U16 = 2,
        S16 = 3,
        F16 = 4,
        U32 = 5,
        S32 = 6,
        F32 = 7,
        U64 = 8,
        S64 = 9,
        F64 = 10
    };
#define STATIC_SHAPE 32
// If this flag is thrown then the channel's of an image are packed into the last dimension of the tensor
// thus H x W x C which is opencv's format instead of C x H x W which is caffe's format.
#define CHANNEL_LAST_DIM 64
// Throw this flag if the number of dimensions of the tensor are statically defined
#define STATIC_DIMS 128
#define CONTINUOUS_TENSOR 256


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
            for(int i = dims - 1; i >= 0; --i)
            {
                size *= stride[i];
            }
            return size;
        }
        
        template<typename T> T& at(const std::initializer_list<int>& idx)
        {
            size_t offset = 0;
            int i = 0;
            for(auto itr = idx.begin(); itr != idx.end(); ++itr, ++i)
            {
                offset += *itr * stride[i];
            }
            return h_dataStart + offset;
        }
        template<typename T, int N> T& at(int indecies[N])
        {
            size_t offset = 0;
            for(int i = N-1; i >= 0; --i)
            {
                offset += indecies[i] * stride[i];
            }
            return *(static_cast<T*>(h_dataStart + offset));
        }

        SyncedMemory* data;
        size_t* stride;
        size_t* shape;
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
            this->shape = this->_shape;
            this->stride = this->_stride;
            this->dims = Shape::N;
            this->flags = Shape::_flags;
            this->flags += DataType<T>::DType;
            if(Shape::_flags & STATIC_SIZE)
            {
                Allocator::getDefaultAllocator()->Allocate(this, NumBytes(), DataType<T>::size);
                h_dataStart = h_data;
                h_dataEnd = h_dataStart + NumBytes();
            }
        }

        template<typename ... ArgTypes>
        T& at(ArgTypes... args)
        {
            size_t offset = 0;
            Offset(offset, 0, this->stride, args...);
            return *(T*)(h_dataStart + offset);
        }
        template<typename ... ArgTypes>
        const T& at(ArgTypes... args) const
        {
            size_t offset = 0;
            Offset(offset, 0, this->stride, args...);
            return *(T*)(h_dataStart + offset);
        }

        template<typename ... ArgTypes>
        T& operator()(ArgTypes... args)
        {
            return this->at(args...);
        }
        template<typename ... ArgTypes>
        const T& operator()(ArgTypes... args) const
        {
            return this->at(args...);
        }

        T& at(const std::initializer_list<int>& idx)
        {
            assert(idx.size() <= this->dims);
            size_t offset = 0;
            int i = 0;
            for (auto itr = idx.begin(); itr != idx.end(); ++itr, ++i)
            {
                offset += *itr * stride[i];
            }
            return *(T*)(h_dataStart + offset);
        }

        const T& at(const std::initializer_list<int>& idx) const
        {
            assert(idx.size() <= this->dims);
            size_t offset = 0;
            int i = 0;
            for (auto itr = idx.begin(); itr != idx.end(); ++itr, ++i)
            {
                offset += *itr * stride[i];
            }
            return *(T*)(h_dataStart + offset);
        }
    };

}
