#pragma once
#include <stdint.h>
#include "Defs.hpp"
#include <vector_types.h>
namespace tcv
{
template<class T> struct DataType {};
uint8_t typeToSize(uint8_t type);

template<> struct DataType<uint8_t>
{
    static const uint8_t DType = U8;
    static const int channels = 1;
    static const int size = sizeof(uint8_t);
};
template<> struct DataType<int8_t>
{
    static const uint8_t DType = S8;
    static const int channels = 1;
    static const int size = sizeof(int8_t);
};
template<> struct DataType<uint16_t>
{
    static const uint8_t DType = U16;
    static const int channels = 1;
    static const int size = sizeof(uint16_t);
};
template<> struct DataType<int16_t>
{
    static const uint8_t DType = S16;
    static const int channels = 1;
    static const int size = sizeof(int16_t);
};
template<> struct DataType<uint32_t>
{
    static const uint8_t DType = U32;
    static const int channels = 1;
    static const int size = sizeof(uint32_t);
};
template<> struct DataType<int32_t>
{
    static const uint8_t DType = S32;
    static const int channels = 1;
    static const int size = sizeof(int32_t);
};
template<> struct DataType<float>
{
    static const uint8_t DType = F32;
    static const int channels = 1;
    static const int size = sizeof(float);
};
template<> struct DataType<uint64_t>
{
    static const uint8_t DType = U64;
    static const int channels = 1;
    static const int size = sizeof(uint64_t);
};
template<> struct DataType<int64_t>
{
    static const uint8_t DType = S64;
    static const int channels = 1;
    static const int size = sizeof(int64_t);
};
template<> struct DataType<double>
{
    static const uint8_t DType = F64;
    static const int channels = 1;
    static const int size = sizeof(double);
};

#define EXPAND_CUDA_VECTOR_TYPE(base, flag) \
template<> struct DataType<base##1> \
{ \
    static const uint8_t DType = flag; \
    static const int channels = 1; \
    static const int size = sizeof(base##1); \
}; \
template<> struct DataType<base##2> \
{ \
    static const uint8_t DType = flag; \
    static const int channels = 2; \
    static const int size = sizeof(base##1); \
}; \
template<> struct DataType<base##3> \
{ \
    static const uint8_t DType = flag; \
    static const int channels = 3; \
    static const int size = sizeof(base##1); \
}; \
template<> struct DataType<base##4> \
{ \
    static const uint8_t DType = flag; \
    static const int channels = 4; \
    static const int size = sizeof(base##1); \
}

EXPAND_CUDA_VECTOR_TYPE(uchar, U8);
EXPAND_CUDA_VECTOR_TYPE(char, S8);
EXPAND_CUDA_VECTOR_TYPE(ushort, U16);
EXPAND_CUDA_VECTOR_TYPE(short, S16);
EXPAND_CUDA_VECTOR_TYPE(uint, U32);
EXPAND_CUDA_VECTOR_TYPE(int, S32);
EXPAND_CUDA_VECTOR_TYPE(float, F32);
EXPAND_CUDA_VECTOR_TYPE(double, F64);
EXPAND_CUDA_VECTOR_TYPE(long, S32);
EXPAND_CUDA_VECTOR_TYPE(ulong, U32);
EXPAND_CUDA_VECTOR_TYPE(longlong, S64);
EXPAND_CUDA_VECTOR_TYPE(ulonglong, U64);


} // namespace tcv
