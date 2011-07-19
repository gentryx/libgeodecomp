#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_gridvecconv_h_
#define _libgeodecomp_parallelization_hiparsimulator_gridvecconv_h_

#include <libgeodecomp/misc/region.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

class GridVecConv
{
public:
    template<typename CELL_TYPE, typename GRID_TYPE, int DIM>
    static SuperVector<CELL_TYPE> gridToVector(
        const GRID_TYPE& sourceGrid, 
        const Region<DIM>& region)
    {
        unsigned size = 0;
        // fixme: cache region size directly in region?!<2
        for (StreakIterator<DIM> i = region.beginStreak(); 
             i != region.endStreak(); ++i) 
            size += i->length();
        SuperVector<CELL_TYPE> res(size);
        CELL_TYPE *dest = &res[0];

        for (StreakIterator<DIM> i = region.beginStreak(); 
             i != region.endStreak(); ++i) {
            const CELL_TYPE *start = &sourceGrid[i->origin];
            std::copy(start, start + i->length(), dest);
            dest += i->length();
        }

        return res;
    }

    // fixme: check that region size == stored vector size
    template<typename CELL_TYPE, typename GRID_TYPE, int DIM>
    static void vectorToGrid(
        const SuperVector<CELL_TYPE>& sourceVector, 
        GRID_TYPE& destGrid, 
        const Region<DIM>& region)
    {
        const CELL_TYPE *source = &sourceVector[0];
        for (StreakIterator<DIM> i = region.beginStreak(); 
             i != region.endStreak(); ++i) {
            unsigned length = i->length();
            const CELL_TYPE *end = source + length;
            std::copy(source, end, &destGrid[i->origin]);
            source = end;
        }
        if (source != &sourceVector.back() + 1)
            throw std::logic_error("region doesn't match vector size");
    }
};

}
}

#endif
#endif
