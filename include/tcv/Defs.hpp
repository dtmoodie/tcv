#pragma once

#define STATIC_SHAPE 32
// If this flag is thrown then the channel's of an image are packed into the last dimension of the tensor
// thus H x W x C which is opencv's format instead of C x H x W which is caffe's format.
#define CHANNEL_LAST_DIM 64
// Throw this flag if the number of dimensions of the tensor are statically defined
#define STATIC_DIMS 128
#define CONTINUOUS_TENSOR 256

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


}
