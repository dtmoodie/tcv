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
    enum{
        DType = U8,
        channels = 1,
        size = sizeof(uint8_t)
    };
};
template<> struct DataType<int8_t>
{
    enum{
        DType = S8,
        channels = 1,
        size = sizeof(int8_t)
    };
};
template<> struct DataType<uint16_t>
{
    enum{
        DType = U16,
        channels = 1,
        size = sizeof(uint16_t)
    };
};
template<> struct DataType<int16_t>
{
    enum{
        DType = S16,
        channels = 1,
        size = sizeof(int16_t)
    };
};
template<> struct DataType<uint32_t>
{
    enum{
        DType = U32,
        channels = 1,
        size = sizeof(uint32_t)
    };
};
template<> struct DataType<int32_t>
{
    enum{
        DType = S32,
        channels = 1,
        size = sizeof(int32_t)
    };
};
template<> struct DataType<float>
{
    enum{
        DType = F32,
        channels = 1,
        size = sizeof(float)
    };
};
template<> struct DataType<uint64_t>
{
    enum{
        DType = U64,
        channels = 1,
        size = sizeof(uint64_t)
    };
};
template<> struct DataType<int64_t>
{
    enum{
        DType = S64,
        channels = 1,
        size = sizeof(int64_t)
    };
};
template<> struct DataType<double>
{
    enum{
        DType = F64,
        channels = 1,
        size = sizeof(double)
    };
};

#define EXPAND_CUDA_VECTOR_TYPE(base, flag) \
template<> struct DataType<base##1> \
{ \
    enum{ \
        DType = flag, \
        channels = 1, \
        size = sizeof(base##1) \
    }; \
}; \
template<> struct DataType<base##2> \
{ \
    enum{ \
        DType = flag, \
        channels = 2, \
        size = sizeof(base##1) \
    }; \
}; \
template<> struct DataType<base##3> \
{ \
    enum{ \
        DType = flag, \
        channels = 3, \
        size = sizeof(base##1) \
    }; \
}; \
template<> struct DataType<base##4> \
{ \
    enum{ \
        DType = flag, \
        channels = 4, \
        size = sizeof(base##1) \
    }; \
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
