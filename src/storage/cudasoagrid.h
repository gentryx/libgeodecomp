#ifndef LIBGEODECOMP_STORAGE_CUDASOAGRID_H
#define LIBGEODECOMP_STORAGE_CUDASOAGRID_H

#include <libflatarray/cuda_array.hpp>

#include <libgeodecomp/geometry/topologies.h>
#include <libgeodecomp/storage/gridbase.h>
#include <libgeodecomp/storage/soagrid.h>

namespace LibGeoDecomp {

/**
 * CUDA-enabled implementation of SoAGrid, relies on
 * LibFlatArray::soa_grid.
 */
template<typename CELL,
         typename TOPOLOGY=Topologies::Cube<2>::Topology,
         bool TOPOLOGICALLY_CORRECT=false>
class CUDASoAGrid : public GridBase<CELL, TOPOLOGY::DIM>
{
public:
    friend class CUDASoAGridTest;

    const static int DIM = TOPOLOGY::DIM;

    using typename GridBase<CELL, DIM>::BufferType;

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
    {}

    void set(const Coord<DIM>&, const CELL&)
    {
        // fixme
    }

    void set(const Streak<DIM>&, const CELL*)
    {
        // fixme
    }

    CELL get(const Coord<DIM>&) const
    {
        // fixme
        return CELL();
    }

    void get(const Streak<DIM>&, CELL *) const
    {
        // fixme
    }

    void setEdge(const CELL&)
    {
        // fixme
    }

    const CELL& getEdge() const
    {
        // fixme
    }

    CoordBox<DIM> boundingBox() const
    {
        // fixme
        return CoordBox<DIM>();
    }


    void saveRegion(BufferType *buffer, const Region<DIM>& region, const Coord<DIM>& offset = Coord<DIM>()) const
    {
        // fixme
    }

    void loadRegion(const BufferType& buffer, const Region<DIM>& region, const Coord<DIM>& offset = Coord<DIM>())
    {
        // fixme
    }

protected:
    virtual void saveMemberImplementation(
        char *target,
        MemoryLocation::Location targetLocation,
        const Selector<CELL>& selector,
        const Region<DIM>& region) const
    {
        // fixme
    }

    virtual void loadMemberImplementation(
        const char *source,
        MemoryLocation::Location sourceLocation,
        const Selector<CELL>& selector,
        const Region<DIM>& region)
    {
        // fixme
    }

private:
    Coord<3> edgeRadii;
    CELL edgeCell;
    CoordBox<DIM> box;
    Coord<DIM> topoDimensions;

    static Coord<3> calcEdgeRadii()
    {
        return SoAGrid<CELL>::calcEdgeRadii();
    }
};

}

#endif
