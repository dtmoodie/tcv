#include "tcv/TypeTraits.hpp"

#define HANDLE_TYPE(TYPE) case DataType<TYPE>::DType: return DataType<TYPE>::size
uint8_t tcv::typeToSize(uint8_t type)
{
    switch(type)
    {
        HANDLE_TYPE(uint8_t);
        HANDLE_TYPE(int8_t);
        HANDLE_TYPE(uint16_t);
        HANDLE_TYPE(int16_t);
        HANDLE_TYPE(uint32_t);
        HANDLE_TYPE(int32_t);
        HANDLE_TYPE(float);
        HANDLE_TYPE(uint64_t);
        HANDLE_TYPE(int64_t);
        HANDLE_TYPE(double);
    }
}
