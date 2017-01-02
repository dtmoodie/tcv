#pragma once
#include "yar/Logging.hpp"
#include "TypeTraits.hpp"
#include <opencv2/core/matx.hpp>
namespace tcv
{
    template<int I, int64_t N, int64_t... Dims> struct DimHandle : public DimHandle<I + 1, Dims...>
    {
        template<class...Args>
        DimHandle(Args... args) :
            DimHandle<I + 1, Dims...>(args...)
        {
        }
        DimHandle():
            DimHandle<I + 1, Dims...>()
        {
        
        }
        template<class Arg, class...Args>
        constexpr static size_t getOffset(Arg arg, Args... args)
        {
            return DimHandle<I + 1, Dims...>::getOffset(args...) + arg*getStride(I);
        }

        constexpr static size_t getStride(int dim)
        {
            return dim == I ?
                DimHandle<I + 1, Dims...>::getSize(dim + 1) :
                DimHandle<I + 1, Dims...>::getStride(dim);
        }

        constexpr static size_t getSize(int dim)
        {
            return dim == I ?
                DimHandle<I + 1, Dims...>::getSize(dim + 1)*N :
                DimHandle<I + 1, Dims...>::getSize(dim);
        }
        constexpr static size_t getShape(int dim)
        {
            return dim == I ?
                N :
                DimHandle<I + 1, Dims...>::getShape(dim);
        }
        constexpr static size_t getStaticShape(int dim)
        {
            return dim == I ?
                N :
                DimHandle<I + 1, Dims...>::getShape(dim);
        }
        void setDimSize(int dim, size_t n)
        {
            DimHandle<I + 1, Dims...>::setDimSize(dim, n);
        }
        template<class... Args>
        void setShape(Args... args)
        {
            DimHandle<I + 1, Dims...>::setShape(args...);
        }
    };

    template<int I, int64_t N> struct DimHandle<I, N>
    {
        template<class Arg>
        DimHandle(Arg arg)
        {
        }
        DimHandle()
        {
        }
        constexpr static size_t getOffset()
        {
            return 0;
        }
        template<class Arg>
        constexpr static size_t getOffset(Arg arg)
        {
            return arg;
        }
        constexpr static size_t getStride(int dim)
        {
            return 1;
        }
        constexpr static size_t getSize(int dim)
        {
            return N;
        }
        constexpr static size_t getShape(int dim)
        {
            return N;
        }
        constexpr static size_t getStaticShape(int dim)
        {
            return N;
        }
        void setDimSize(int dim, size_t n)
        {
            return;
        }
        template<class...Args>
        void setShape(Args... args)
        {
        }
        template<class U>
        void setShape(U arg)
        {
            
        }
    };
    template<int I, int64_t... Dims> 
    struct DimHandle<I, -1, Dims...> 
        : public DimHandle<I + 1, Dims...>
    {
        template<class U, class...Args>
        DimHandle(U arg, Args... args) :
            N(arg),
            DimHandle<I + 1, Dims...>(args...)
        {
        }
        DimHandle():
            N(0),
            DimHandle<I + 1, Dims...>()
        {
        }
        template<class Arg, class... Args>
        size_t getOffset(Arg arg, Args... args) const
        {
            return DimHandle<I + 1, Dims...>::getOffset(args...) + arg*getStride(I);
        }
        
        size_t getStride(int dim) const
        {
            return dim == I ?
                DimHandle<I + 1, Dims...>::getSize(dim + 1):
                DimHandle<I + 1, Dims...>::getStride(dim);
        }
        size_t getSize(int dim) const
        {
            return dim == I ?
                DimHandle<I + 1, Dims...>::getSize(dim + 1)*N :
                DimHandle<I + 1, Dims...>::getSize(dim);
        }
        size_t getShape(int dim) const
        {
            return dim == I ?
                N :
                DimHandle<I + 1, Dims...>::getShape(dim);
        }
        constexpr static size_t getStaticShape(int dim)
        {
            return dim == I ?
                0 :
                DimHandle<I + 1, Dims...>::getStaticShape(dim);
        }
        void setDimSize(int dim, size_t N_)
        {
            if (dim == I)
                N = N_;
            else
                DimHandle<I + 1, Dims...>::setDimSize(dim, N_);
        }
        template<class U, class...Args>
        void setShape(U arg, Args... args)
        {
            N = arg;
            DimHandle<I + 1, Dims...>::setShape(args...);
        }
        void setShape()
        {
            
        }
    protected:
        size_t N;
    };

    template<int I> struct DimHandle<I, -1>
    {
        template<class U>
        DimHandle(U arg) :
            N(arg)
        {
        }
        DimHandle():
            N(0)
        {
        }
        template<class Arg>
        constexpr static size_t getOffset(Arg arg)
        {
            return arg;
        }
        constexpr static size_t getStride(int dim)
        {
            return 1;
        }
        size_t getSize(int dim) const
        {
            return N;
        }
        size_t getShape(int dim) const
        {
            return N;
        }
        constexpr static size_t getStaticShape(int dim)
        {
            return 0;
        }
        void setDimSize(int dim, size_t N_)
        {
            if (dim == I)
                N = N_;
        }
        template<class U>
        void setShape(U arg)
        {
            N = arg;
        }
    protected:
        size_t N;
    };

    template<class T, int64_t... Size>
    struct StaticShape : protected DimHandle<0, Size..., DataType<T>::channels>
    {
        typedef DimHandle<0, Size..., DataType<T>::channels> DimType;
        template<class... Args>
        StaticShape(SyncedMemory_<size_t>& shape, SyncedMemory_<size_t>& stride, Args... args) :
            DimType(args...)
        {
            size_t size = DataType<T>::size; //sizeof(T);
            int N = getNumDims();
            shape.resize(getNumDims());
            stride.resize(getNumDims());
            _shape = shape.getCpuMutable();
            _stride = stride.getCpuMutable();
            setShape(args...);
        }
        StaticShape(SyncedMemory_<size_t>* shape, SyncedMemory_<size_t>* stride) :
            DimType()
        {
            size_t size = DataType<T>::size;
            int N = getNumDims();
            ASSERT_EQ(shape->getSize(), getNumDims());
            ASSERT_EQ(stride->getSize(), getNumDims());
            _shape = shape->getCpuMutable();
            _stride = stride->getCpuMutable();
            for(int i = 0; i < N; ++i)
            {
                DimType::setDimSize(i, _shape[i] / DataType<T>::size);
            }    
        }

        template<class...Args>
        size_t getOffset(Args... args)
        {
            return DimType::getOffset(args...);
        }
        constexpr size_t getStride(int dim = 0)
        {
            return DimType::getStride(dim);
        }
        constexpr size_t getStrideBytes(int dim = 0)
        {
            return DimType::getStride(dim) * DataType<T>::size;
        }

        constexpr size_t getNumBytes(int dim = 0)
        {
            return DimType::getSize(dim) * DataType<T>::size;
        }

        constexpr size_t getNumElements(int dim = 0)
        {
            return DimType::getSize(dim);
        }
        constexpr size_t getShape(int dim = 0)
        {
            return DimType::getShape(dim);
        }
        constexpr static size_t getStaticShape(int dim = 0)
        {
            return DimType::getStaticShape(dim);
        }
        constexpr size_t getShapeBytes(int dim = 0)
        {
            return DimType::getShape(dim) * DataType<T>::size;
        }
        constexpr static size_t getNumDims()
        {
            return DataType<T>::channels == 1 ? 
                sizeof...(Size) :
                sizeof...(Size) + 1;
        }
        constexpr static int shapeFlags()
        {
            return STATIC_SHAPE + STATIC_DIMS;
        }
        template<class...Args>
        void setShape(Args... args)
        {
            DimType::setShape(args...);
            for(int i = 0; i < getNumDims(); ++i)
            {
                _shape[i] = getShapeBytes(i);
                _stride[i] = getStrideBytes(i);
            }
        }
        void setDimSize(int dim, size_t size)
        {
            DimType::setDimSize(dim, size);
            for (int i = 0; i < getNumDims(); ++i)
            {
                _shape[i] = getShapeBytes(i);
                _stride[i] = getStrideBytes(i);
            }
        }
    protected:
        size_t* _shape;
        size_t* _stride;
    };
}
