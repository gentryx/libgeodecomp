#ifndef _libgeodecomp_parallelization_hiparsimulator_gridvecconv_h_
#define _libgeodecomp_parallelization_hiparsimulator_gridvecconv_h_

#include <libgeodecomp/config.h>
#include <libgeodecomp/misc/region.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

class GridVecConv
{
public:
    template<typename GRID_TYPE>
    static SuperVector<typename GRID_TYPE::CellType> gridToVector(
        const GRID_TYPE& grid, 
        const Region<GRID_TYPE::DIM>& region)
    {
        SuperVector<typename GRID_TYPE::CellType> ret(region.size());
        gridToVector(grid, &ret, region);
        return ret;
    }

    template<typename GRID_TYPE>
    static void gridToVector(
        const GRID_TYPE& grid, 
        SuperVector<typename GRID_TYPE::CellType> *vec,
        const Region<GRID_TYPE::DIM>& region)
    {
        if (vec->size() != std::size_t(region.size()))
            throw std::logic_error("region doesn't match vector size");

        typename GRID_TYPE::CellType *dest = &(*vec)[0];
        
        for (typename Region<GRID_TYPE::DIM>::StreakIterator i = region.beginStreak(); 
             i != region.endStreak(); ++i) {
            const typename GRID_TYPE::CellType *start = &grid[i->origin];
            std::copy(start, start + i->length(), dest);
            dest += i->length();
        }
    }

    template<typename VEC_TYPE, typename GRID_TYPE, typename REGION_TYPE>
    static void vectorToGrid(
        const VEC_TYPE& vec, 
        GRID_TYPE *grid, 
        const REGION_TYPE& region)
    {
        if (vec.size() != std::size_t(region.size()))
            throw std::logic_error("region doesn't match vector size");
        
        if(vec.size() == 0) return;

        const typename GRID_TYPE::CellType *source = &vec[0];
        for (typename Region<GRID_TYPE::DIM>::StreakIterator i = region.beginStreak(); 
             i != region.endStreak(); ++i) {
            unsigned length = i->length();
            const typename GRID_TYPE::CellType *end = source + length;
            typename GRID_TYPE::CellType *dest = &(*grid)[i->origin];
            std::copy(source, end, dest);
            source = end;
        }
    }
};

}
}

#endif
