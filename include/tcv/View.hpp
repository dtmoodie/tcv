#include "Tensor.hpp"

namespace tcv
{
    class GPU;
    class CPU;

    template<class XPU = CPU> class View{};

    template<> class View<GPU>
    {
    public:
        View(int dim, Tensor& tensor_)
        {
            dims = tensor_.dims - dim;
            shape = tensor_.shape->getGpu() + dim;
            start = tensor_.data->getGpuMutable();
            stride = tensor_.stride->getGpu() + dim;
            end = start + tensor_.data->getSize();
        }

        template<typename ... ArgTypes>
        uint8_t& operator()(ArgTypes... args)
        {
            return start + offset(stride, dims, args...);
        }
    protected:
        uint8_t* start;
        uint8_t* end;
        const size_t* shape;
        const size_t* stride;
        int dims;
    };
    
    template<> class View<CPU>
    {
    public:
        View(int dim, Tensor& tensor_)
        {
            dims = tensor_.dims - dim;
            shape = tensor_.shape->getCpu() + dim;
            start = tensor_.data->getCpuMutable();
            stride = tensor_.stride->getCpu() + dim;
            end = start + tensor_.data->getSize();
        }

        template<typename ... ArgTypes>
        uint8_t& operator()(ArgTypes... args)
        {
            return start + offset(stride, dims, args...);
        }
    protected:
        uint8_t* start;
        uint8_t* end;
        const size_t* shape;
        const size_t* stride;
        int dims;
    };

    template<class XPU> class ConstView
    {
    public:
        ConstView(int dim, const Tensor& tensor_)
        {
            dims = tensor_.dims - dim;
            shape = tensor_.shape->getCpu() + dim;
            start = tensor_.data->getCpu();
            stride = tensor_.stride->getCpu() + dim;
            end = start + tensor_.data->getSize();
        }

        template<typename ... ArgTypes>
        uint8_t& operator()(ArgTypes... args)
        {
            return start + offset(stride, dims, args...);
        }
    protected:
        const uint8_t* start;
        const uint8_t* end;
        const size_t* shape;
        const size_t* stride;
        int dims;
    };

    template<class T, class XPU = CPU>
    class View_ : protected View<XPU>
    {
    public:
        View_(int dim, Tensor& tensor_):
            View(dim, tensor_)
        {
        }
        template<typename ... ArgTypes>
        T& operator()(ArgTypes... args)
        {
            size_t offset_ = offset(stride, dims, args...);
            offset_ /= sizeof(T);
            return *((T*)start + offset_);
        }
        template<typename ... ArgTypes>
        const T& operator()(ArgTypes... args) const
        {
            size_t offset_ = offset(stride, dims, args...);
            offset_ /= sizeof(T);
            return *((T*)start + offset_);
        }
        size_t size(int dim = -1) const
        {
            if(dim == -1)
            {
                return getSize(this->shape, dims) / sizeof(T);
            }
            
            return this->shape[dim] / sizeof(T);
        }
    };
    
    template<class T, class XPU = CPU>
    class ConstView_ : protected ConstView<XPU>
    {
    public:
        ConstView_(int dim, const Tensor& tensor_) :
            ConstView(dim, tensor_)
        {
        }
        template<typename ... ArgTypes>
        const T& operator()(ArgTypes... args) const
        {
            size_t offset_ = offset(stride, dims, args...);
            offset_ /= sizeof(T);
            return *((T*)start + offset_);
        }
        size_t getSize(int dim = -1) const
        {
            if (dim == -1)
            {
                return getSize(this->shape, dims) / sizeof(T);
            }

            return this->shape[dim] / sizeof(T);
        }
    };
}