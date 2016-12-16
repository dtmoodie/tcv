#include <tcv/Tensor.hpp>


int main()
{
    tcv::Tensor_<float, tcv::StaticShape<float, 4, 5, 3>> Mat4x5x3;
    Mat4x5x3.at({0,1,2});
    size_t offset = 0;
    size_t stride[2] = {10, 2};
    Mat4x5x3.at(0,1,2) = 1;
    float val = Mat4x5x3.at(0,1,2);
    Mat4x5x3.at(0,1,2) *= 2;
    float val2 = Mat4x5x3(0,1,2);
    return 0;
}
