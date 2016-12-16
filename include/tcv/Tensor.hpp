#pragma once
#include <stdint.h>
#include <initializer_list>
#include <type_traits>
#include <cassert>
#include <cstddef>

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
#define STATIC_SIZE 32

    template<class T> struct DataType{};
    template<> struct DataType<uint8_t>
    {
        static const short DType = U8;
        static const int channels = 1;
        static const int size = sizeof(uint8_t);
    };
    template<> struct DataType<int8_t>
    {
        static const short DType = S8;
        static const int channels = 1;
        static const int size = sizeof(int8_t);
    };
    template<> struct DataType<uint16_t>
    {
        static const short DType = U16;
        static const int channels = 1;
        static const int size = sizeof(uint16_t);
    };
    template<> struct DataType<int16_t>
    {
        static const short DType = S16;
        static const int channels = 1;
        static const int size = sizeof(int16_t);
    };
    template<> struct DataType<uint32_t>
    {
        static const short DType = U32;
        static const int channels = 1;
        static const int size = sizeof(uint32_t);
    };
    template<> struct DataType<int32_t>
    {
        static const short DType = S32;
        static const int channels = 1;
        static const int size = sizeof(int32_t);
    };
    template<> struct DataType<float>
    {
        static const short DType = F32;
        static const int channels = 1;
        static const int size = sizeof(float);
    };
    template<> struct DataType<uint64_t>
    {
        static const short DType = U64;
        static const int channels = 1;
        static const int size = sizeof(uint64_t);
    };
    template<> struct DataType<int64_t>
    {
        static const short DType = S64;
        static const int channels = 1;
        static const int size = sizeof(int64_t);
    };
    template<> struct DataType<double>
    {
        static const short DType = F64;
        static const int channels = 1;
        static const int size = sizeof(double);
    };
    
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
    class Tensor
    {
    public:
        class Allocator
        {
        public:
            virtual ~Allocator(){}
            virtual bool Allocate(Tensor* tensor, size_t bytes, int elemSize) = 0;
            virtual bool Deallocate(Tensor* tensor) = 0;
            static Allocator* getDefaultAllocator();
        };
        Tensor()
        {
            h_data = nullptr;
            h_dataStart = nullptr;
            h_dataEnd = nullptr;
            d_data = nullptr;
            d_dataStart = nullptr;
            d_dataEnd = nullptr;
            stride = nullptr;
            shape = nullptr;
            dims = 0;
            allocator = nullptr;
            flags = 0;
        }
        ~Tensor()
        {
            if(refCount)
            {
                --*refCount;
                if(*refCount == 0)
                {
                    if(allocator)
                    {
                        if(!allocator->Deallocate(this))
                        {
                            Allocator::getDefaultAllocator()->Deallocate(this);
                        }
                    }
                }
            }
        }
        size_t NumBytes() const
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

        // Points to the head of the data regardless of the view
        uint8_t* h_data;
        // Beginning of the view
        uint8_t* h_dataStart;
        // End of the view
        uint8_t* h_dataEnd;

        uint8_t* d_data;
        uint8_t* d_dataStart;
        uint8_t* d_dataEnd;
        size_t* stride;
        size_t* shape;
        int dims;
        Allocator* allocator;
        uint64_t flags;
        int* refCount;
        void cleanup()
        {
            if(allocator)
            {
                allocator->Deallocate(this);
            }
        }
    };

    template<class T, int N> struct Shape
    {
        size_t _shape[N];
        size_t _stride[N];
    };
    template<class T> struct Shape<T, -1>
    {
        
    };
    template<class T, int... Size> struct StaticShape
    {
        static const int N = sizeof...(Size);
        static const int _flags = STATIC_SIZE;
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
    
    class View
    {
    public:

    private:
        uint8_t* begin;
        uint8_t* end;
        size_t* stride;
        size_t* shape;
        int dims;
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
