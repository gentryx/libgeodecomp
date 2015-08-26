#ifndef LIBGEODECOMP_STORAGE_CUDASOAGRID_H
#define LIBGEODECOMP_STORAGE_CUDASOAGRID_H

#include <libgeodecomp/geometry/topologies.h>
#include <libgeodecomp/storage/cudaarray.h>
#include <libgeodecomp/storage/gridbase.h>
#include <libgeodecomp/storage/soagrid.h>

namespace LibGeoDecomp {

/**
 * CUDA-enabled implementation of SoAGrid, relies on
 * LibFlatArray::soa_grid.
 */
template<typename CELL_TYPE,
         typename TOPOLOGY=Topologies::Cube<2>::Topology,
         bool TOPOLOGICALLY_CORRECT=false>
class CUDASoAGrid : public GridBase<CELL, TOPOLOGY::DIM>
{
public:
    friend class CUDASoAGridTest;

    const static int DIM = TOPOLOGY::DIM;
    /**
     * Accumulated size of all members. Note that this may be lower
     * than sizeof(CELL) as the compiler may add padding within a
     * struct/class to ensure alignment.
     */
    static const int AGGREGATED_MEMBER_SIZE =  LibFlatArray::aggregated_member_size<CELL>::VALUE;

    typedef CELL CellType;
    typedef TOPOLOGY Topology;
    typedef LibFlatArray::soa_grid<CELL> Delegate;
    typedef typename APITraits::SelectStencil<CELL>::Value Stencil;

    explicit CUDASoAGrid(
        const CoordBox<DIM>& box = CoordBox<DIM>(),
        const CELL& defaultCell = CELL(),
        const CELL& edgeCell = CELL(),
        const Coord<DIM>& topologicalDimensions = Coord<DIM>()) :
        edgeRadii(calcEdgeRadii()),
        edgeCell(edgeCell),
        box(box),
        topoDimensions(topologicalDimensions)
    {
    }

};

}

#endif
