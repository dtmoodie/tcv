#include "Tensor.hpp"

namespace tcv
{
    class View : public Tensor
    {
    public:


    protected:
        size_t* start;
        size_t* end;
    };
}