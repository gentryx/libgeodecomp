#ifndef LIBGEODECOMP_STORAGE_CUDASOAGRID_H
#define LIBGEODECOMP_STORAGE_CUDASOAGRID_H

#include <libflatarray/cuda_array.hpp>

#include <libgeodecomp/geometry/topologies.h>
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
    using GridBase<CELL, DIM>::topoDimensions;

    /**
     * Accumulated size of all members. Note that this may be lower
     * than sizeof(CELL) as the compiler may add padding within a
     * struct/class to ensure alignment.
     */
    static const int AGGREGATED_MEMBER_SIZE =  LibFlatArray::aggregated_member_size<CELL>::VALUE;

    typedef CELL CellType;
    typedef TOPOLOGY Topology;
    typedef LibFlatArray::soa_grid<CELL, LibFlatArray::cuda_allocator<char>, true> Delegate;
    typedef typename APITraits::SelectStencil<CELL>::Value Stencil;

    explicit CUDASoAGrid(
        const CoordBox<DIM>& box = CoordBox<DIM>(),
        const CELL& defaultCell = CELL(),
        const CELL& edgeCell = CELL(),
        const Coord<DIM>& topologicalDimensions = Coord<DIM>()) :
        GridBase<CELL, DIM>(topologicalDimensions),
        edgeRadii(calcEdgeRadii()),
        edgeCell(edgeCell),
        box(box)
    {
        actualDimensions = Coord<3>::diagonal(1);
        for (int i = 0; i < DIM; ++i) {
            actualDimensions[i] = box.dimensions[i];
        }
        actualDimensions += edgeRadii * 2;

        delegate.resize(
            actualDimensions.x(),
            actualDimensions.y(),
            actualDimensions.z());

    }

    void set(const Coord<DIM>& absoluteCoord, const CELL& cell)
    {
        Coord<DIM> relativeCoord = absoluteCoord - box.origin;
        if (TOPOLOGICALLY_CORRECT) {
            relativeCoord = Topology::normalize(relativeCoord, topoDimensions);
        }
        if (Topology::isOutOfBounds(relativeCoord, box.dimensions)) {
            setEdge(cell);
            return;
        }

        delegateSet(relativeCoord, cell);
    }

    void set(const Streak<DIM>& streak, const CELL* cells)
    {
        Coord<DIM> relativeCoord = streak.origin - box.origin;
        if (TOPOLOGICALLY_CORRECT) {
            relativeCoord = Topology::normalize(relativeCoord, topoDimensions);
        }

        delegateSet(relativeCoord, cells, streak.length());
    }

    CELL get(const Coord<DIM>& absoluteCoord) const
    {
        Coord<DIM> relativeCoord = absoluteCoord - box.origin;
        if (TOPOLOGICALLY_CORRECT) {
            relativeCoord = Topology::normalize(relativeCoord, topoDimensions);
        }
        if (Topology::isOutOfBounds(relativeCoord, box.dimensions)) {
            return edgeCell;
        }

        return delegateGet(relativeCoord);
    }

    void get(const Streak<DIM>& streak, CELL *cells) const
    {
        Coord<DIM> relativeCoord = streak.origin - box.origin;
        if (TOPOLOGICALLY_CORRECT) {
            relativeCoord = Topology::normalize(relativeCoord, topoDimensions);
        }

        delegateGet(relativeCoord, cells, streak.length());
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
    Delegate delegate;
    Coord<3> edgeRadii;
    Coord<3> actualDimensions;
    CELL edgeCell;
    CoordBox<DIM> box;

    static Coord<3> calcEdgeRadii()
    {
        return SoAGrid<CELL>::calcEdgeRadii();
    }

    CELL delegateGet(const Coord<1>& coord) const
    {
        return delegate.get(
            edgeRadii.x() + coord.x(),
            edgeRadii.y(),
            edgeRadii.z());
    }

    CELL delegateGet(const Coord<2>& coord) const
    {
        return delegate.get(
            edgeRadii.x() + coord.x(),
            edgeRadii.y() + coord.y(),
            edgeRadii.z());
    }

    CELL delegateGet(const Coord<3>& coord) const
    {
        return delegate.get(
            edgeRadii.x() + coord.x(),
            edgeRadii.y() + coord.y(),
            edgeRadii.z() + coord.z());
    }

    void delegateGet(const Coord<1>& coord, CELL *cells, int count) const
    {
        delegate.get(
            edgeRadii.x() + coord.x(),
            edgeRadii.y(),
            edgeRadii.z(),
            cells,
            count);
    }

    void delegateGet(const Coord<2>& coord, CELL *cells, int count) const
    {
        delegate.get(
            edgeRadii.x() + coord.x(),
            edgeRadii.y() + coord.y(),
            edgeRadii.z(),
            cells,
            count);
    }

    void delegateGet(const Coord<3>& coord, CELL *cells, int count) const
    {
        delegate.get(
            edgeRadii.x() + coord.x(),
            edgeRadii.y() + coord.y(),
            edgeRadii.z() + coord.z(),
            cells,
            count);
    }

    void delegateSet(const Coord<1>& coord, const CELL& cell)
    {
        delegate.set(
            edgeRadii.x() + coord.x(),
            edgeRadii.y(),
            edgeRadii.z(),
            cell);
    }

    void delegateSet(const Coord<2>& coord, const CELL& cell)
    {
        delegate.set(
            edgeRadii.x() + coord.x(),
            edgeRadii.y() + coord.y(),
            edgeRadii.z(),
            cell);
    }

    void delegateSet(const Coord<3>& coord, const CELL& cell)
    {
        delegate.set(
            edgeRadii.x() + coord.x(),
            edgeRadii.y() + coord.y(),
            edgeRadii.z() + coord.z(),
            cell);
    }

    void delegateSet(const Coord<1>& coord, const CELL *cells, int count)
    {
        delegate.set(
            edgeRadii.x() + coord.x(),
            edgeRadii.y(),
            edgeRadii.z(),
            cells,
            count);
    }

    void delegateSet(const Coord<2>& coord, const CELL *cells, int count)
    {
        delegate.set(
            edgeRadii.x() + coord.x(),
            edgeRadii.y() + coord.y(),
            edgeRadii.z(),
            cells,
            count);
    }

    void delegateSet(const Coord<3>& coord, const CELL *cells, int count)
    {
        delegate.set(
            edgeRadii.x() + coord.x(),
            edgeRadii.y() + coord.y(),
            edgeRadii.z() + coord.z(),
            cells,
            count);
    }
};

}

#endif
