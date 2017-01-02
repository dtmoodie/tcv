#pragma once
#include <cuda_runtime_api.h>
#include "yar/Logging.hpp"
namespace tcv
{


}

#define ASSERT_OP(op, lhs, rhs) if(!(lhs op rhs)) yar::ThrowOnDestroy(__FUNCTION__, __FILE__, __LINE__).stream() << "[" << #lhs << " " << #op << " " << #rhs << "] - Failed (" << lhs << " " <<  #op << " " << rhs << ")"

#define tcvCudaSafeCall(expr) \
{ \
    cudaError_t error = expr; \
    if(error != cudaSuccess) \
        yar::ThrowOnDestroy(__FUNCTION__, __FILE__, __LINE__).stream() << #expr " failed (" << error << " - " << cudaGetErrorString(error) << ")"; \
}

#define tcvCudaSafeCallStream(expr, msg) \
{ \
    cudaError_t error = expr; \
    if(error != cudaSuccess) \
        yar::ThrowOnDestroy(__FUNCTION__, __FILE__, __LINE__).stream() << #expr " failed (" << error << " - " << cudaGetErrorString(error) << ")" << msg; \
}
