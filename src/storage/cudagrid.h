#ifndef LIBGEODECOMP_STORAGE_CUDAGRID_H
#define LIBGEODECOMP_STORAGE_CUDAGRID_H

#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/storage/cudaarray.h>

#include <cuda.h>

namespace LibGeoDecomp {

/**
 * A lightweight AoS-style (Array of Structs) grid class, to manage
 * memory allocation and bulk data transfer on NVDIA GPUs. It does not
 * implement the GridBase interface as its small-scale data access
 * functions would yield an an unacceptably slow interface.
 */
template<typename CELL_TYPE,
         typename TOPOLOGY=Topologies::Cube<2>::Topology,
         bool TOPOLOGICALLY_CORRECT=false>
class CUDAGrid
{
public:
    friend class CUDAGridTest;

    typedef CELL_TYPE CellType;
    typedef TOPOLOGY Topology;

    static const int DIM = Topology::DIM;

    inline CUDAGrid(
        const CoordBox<DIM>& box = CoordBox<DIM>(),
        const Coord<DIM>& topologicalDimensions = Coord<DIM>()) :
        box(box),
        topoDimensions(topologicalDimensions),
        array(box.dimensions.prod())
    {}

    template<typename GRID_TYPE, typename REGION>
    void saveRegion(GRID_TYPE *target, const REGION& region) const
    {
        for (typename REGION::StreakIterator i = region.beginStreak();
             i != region.endStreak();
             ++i) {
            std::size_t length = i->length() * sizeof(CellType);
            cudaMemcpy(&(*target)[i->origin], const_cast<CellType*>(address(i->origin)), length, cudaMemcpyDeviceToHost);
        }
    }

    template<typename GRID_TYPE, typename REGION>
    void loadRegion(const GRID_TYPE& source, const REGION& region)
    {
        for (typename REGION::StreakIterator i = region.beginStreak();
             i != region.endStreak();
             ++i) {
            std::size_t length = i->length() * sizeof(CellType);
            cudaMemcpy(address(i->origin), &source[i->origin], length, cudaMemcpyHostToDevice);
        }
    }

    CellType *data()
    {
        return array.data();
    }

    const CellType *data() const
    {
        return array.data();
    }

private:
    CoordBox<DIM> box;
    Coord<DIM> topoDimensions;
    CUDAArray<CellType> array;

    std::size_t byteSize() const
    {
        return box.dimensions.prod() * sizeof(CellType);
    }

    std::size_t offset(const Coord<DIM>& absoluteCoord) const
    {
        Coord<DIM> relativeCoord = absoluteCoord - box.origin;
        if (TOPOLOGICALLY_CORRECT) {
            relativeCoord = Topology::normalize(relativeCoord, topoDimensions);
        }

        return relativeCoord.toIndex(box.dimensions);
    }

    CellType *address(const Coord<DIM>& absoluteCoord)
    {
        return data() + offset(absoluteCoord);
    }

    const CellType *address(const Coord<DIM>& absoluteCoord) const
    {
        return data() + offset(absoluteCoord);
    }
};

}

#endif
