#ifndef LIBGEODECOMP_STORAGE_GRIDVECCONV_H
#define LIBGEODECOMP_STORAGE_GRIDVECCONV_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/geometry/region.h>

namespace LibGeoDecomp {

template<typename CELL, typename TOPOLOGY, bool TOPOLOGICALLY_CORRECT>
class SoAGrid;

class GridVecConv
{
public:
    template<typename CELL_TYPE, typename TOPOLOGY_TYPE, bool TOPOLOGICALLY_CORRECT, typename REGION_TYPE>
    static std::vector<CELL_TYPE> gridToVector(
        const DisplacedGrid<CELL_TYPE, TOPOLOGY_TYPE, TOPOLOGICALLY_CORRECT>& grid,
        const REGION_TYPE& region)
    {
        std::vector<CELL_TYPE> ret(region.size());
        gridToVector(grid, &ret, region);
        return ret;
    }

    template<typename CELL_TYPE, typename TOPOLOGY_TYPE, bool TOPOLOGICALLY_CORRECT, typename VECTOR_TYPE, typename REGION_TYPE>
    static void gridToVector(
        const DisplacedGrid<CELL_TYPE, TOPOLOGY_TYPE, TOPOLOGICALLY_CORRECT>& grid,
        VECTOR_TYPE *vec,
        const REGION_TYPE& region)
    {
        if (vec->size() != region.size()) {
            throw std::logic_error("region doesn't match vector size");
        }

        if(vec->size() == 0) {
            return;
        }

        CELL_TYPE *dest = &(*vec)[0];

        for (typename Region<TOPOLOGY_TYPE::DIM>::StreakIterator i = region.beginStreak();
             i != region.endStreak(); ++i) {
            const CELL_TYPE *start = &(grid[i->origin]);
            std::copy(start, start + i->length(), dest);
            dest += i->length();
        }
    }

    template<typename CELL_TYPE, typename TOPOLOGY_TYPE, typename VECTOR_TYPE, bool TOPOLOGICALLY_CORRECT, typename REGION_TYPE>
    static void gridToVector(
        const SoAGrid<CELL_TYPE, TOPOLOGY_TYPE, TOPOLOGICALLY_CORRECT>& grid,
        VECTOR_TYPE *vec,
        const REGION_TYPE& region)
    {
        size_t regionSize = region.size() *
            LibFlatArray::aggregated_member_size<CELL_TYPE>::VALUE;

        if (vec->size() != regionSize) {
            throw std::logic_error("region doesn't match raw vector's size");
        }

        if(vec->size() == 0) {
            return;
        }

        grid.saveRegion(&(*vec)[0], region);
    }

    template<typename VEC_TYPE, typename CELL_TYPE, typename TOPOLOGY_TYPE, bool TOPOLOGICALLY_CORRECT, typename REGION_TYPE>
    static void vectorToGrid(
        const VEC_TYPE& vec,
        DisplacedGrid<CELL_TYPE, TOPOLOGY_TYPE, TOPOLOGICALLY_CORRECT> *grid,
        const REGION_TYPE& region)
    {
        if (vec.size() != region.size()) {
            throw std::logic_error("vector doesn't match region's size");
        }

        if(vec.size() == 0) {
            return;
        }

        const CELL_TYPE *source = &vec[0];

        for (typename REGION_TYPE::StreakIterator i = region.beginStreak();
             i != region.endStreak();
             ++i) {
            unsigned length = i->length();
            const CELL_TYPE *end = source + length;
            CELL_TYPE *dest = &((*grid)[i->origin]);
            std::copy(source, end, dest);
            source = end;
        }
    }

    template<typename VEC_TYPE, typename CELL_TYPE, typename TOPOLOGY_TYPE, typename REGION_TYPE, bool TOPOLOGICALLY_CORRECT>
    static void vectorToGrid(
        const VEC_TYPE& vec,
        SoAGrid<CELL_TYPE, TOPOLOGY_TYPE, TOPOLOGICALLY_CORRECT> *grid,
        const REGION_TYPE& region)
    {
        size_t regionSize = region.size() *
            LibFlatArray::aggregated_member_size<CELL_TYPE>::VALUE;

        if (vec.size() != regionSize) {
            throw std::logic_error("raw vector doesn't match region's size");
        }

        if(vec.size() == 0) {
            return;
        }

        grid->loadRegion(&vec[0], region);
    }

};

}

#endif
