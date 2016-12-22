#pragma once
#include <stdint.h>
#include "Defs.hpp"
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
} // namespace tcv
