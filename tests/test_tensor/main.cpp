#include <tcv/Tensor.hpp>
#include <tcv/SyncedMemoryImpl.hpp>

int main()
{
    tcv::StaticShape<float, 4,5,3> shape;
    auto stride = shape.getStride();
    tcv::Tensor_<float, tcv::StaticShape<float, 4, 5, 3>> Mat4x5x3;
    auto type = Mat4x5x3.elemType();
    Mat4x5x3(0,1,2) = 1;
    float val = Mat4x5x3(0,1,2);
    Mat4x5x3(0,1,2) *= 2;
    float val2 = Mat4x5x3(0,1,2);

    return 0;
}
