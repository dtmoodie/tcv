#include <tcv/Tensor.hpp>
/*
struct expand_type {
    template<typename... T>
    expand_type(T&&...) {}
};
template<typename... ArgTypes>
void Offset(size_t& offset, int index, const size_t* stride, ArgTypes... args)
{
    expand_type{ 0, (offset += args * stride[index], 0)... };
}
*/




int main()
{
    //tcv::Tensor_<float, tcv::StaticShape<float, 4,4>> Mat4x4;
    tcv::Tensor_<float, tcv::StaticShape<float, 4, 5, 3>> Mat4x5x3;
    Mat4x5x3.at({0,1,2});
    size_t offset = 0;
    size_t stride[2] = {10, 2};
    Mat4x5x3.at(0,1,2);

}