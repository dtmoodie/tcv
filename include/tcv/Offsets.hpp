#pragma once

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

} // namespace tcv
