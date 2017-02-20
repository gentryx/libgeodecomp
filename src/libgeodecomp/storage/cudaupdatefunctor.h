#ifndef LIBGEODECOMP_STORAGE_CUDAUPDATEFUNCTOR_H
#define LIBGEODECOMP_STORAGE_CUDAUPDATEFUNCTOR_H

#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_CUDA

namespace LibGeoDecomp {

/**
 * Simpler counterpart to the UpdateFunctor, optimized for CUDA code.
 */
template<typename CELL_TYPE>
class CUDAUpdateFunctor
{
public:
    template<typename HOOD>
    __device__
    void operator()(CELL_TYPE *gridNew, int& index, int offset, const HOOD& hood, const int nanoStep) const
    {
        typedef typename APITraits::SelectSeparateCUDAUpdate<CELL_TYPE>::Value CUDATest;
        CUDATest fixme;
        runUpdate(gridNew, index, offset, hood, nanoStep, fixme);
        index += offset;
    }

private:

    template<typename HOOD>
    __device__
    void runUpdate(CELL_TYPE *gridNew, int& index, int offset, const HOOD& hood, int nanoStep, APITraits::FalseType) const
    {
        gridNew[index].update(hood, nanoStep);
    }

    template<typename HOOD>
    __device__
    void runUpdate(CELL_TYPE *gridNew, int& index, int offset, const HOOD& hood, int nanoStep, APITraits::TrueType) const
    {
        gridNew[index].updateCUDA(hood, nanoStep);
    }
};

}

#endif

#endif
