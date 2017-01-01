#pragma once
#include <opencv2/core/matx.hpp>
#include "TypeTraits.hpp"
namespace tcv
{
    template<class T, int N> 
    struct DataType<cv::Vec<T, N>>
    {
        static const uint8_t DType = DataType<T>::DType;
        static const int channels = N;
        static const int size = sizeof(T);
    };

    template<class T, int R, int C> 
    struct DataType<cv::Matx<T, R, C>>
    {
        static const uint8_t DType = DataType<T>::DType;
        static const int channels = R*C;
        static const int size = sizeof(T);
    };
}