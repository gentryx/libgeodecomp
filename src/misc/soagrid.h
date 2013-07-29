#ifndef LIBGEODECOMP_MISC_SOAGRID_H
#define LIBGEODECOMP_MISC_SOAGRID_H

#include <libflatarray/flat_array.hpp>
#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/gridbase.h>
#include <libgeodecomp/misc/topologies.h>

namespace LibGeoDecomp {

namespace SoAGridHelpers {

template<typename CELL>
class SetEdges
{
public:
    /**
     * Helper class for initializing the edges of a SoAGrid. gridDim
     * refers to the size of the grid as requested by the user.
     * edgeRadii specifies how many layers of cells are to be padded
     * around these, representing the edgeCell. This procedure makes
     * updating the edgeCell slow, but eliminates all conditionals on
     * the grid boundaries. edgeRadii[D] will typically be 0 if the
     * topology wraps this dimension. It'll be equal to the stencil's
     * radius if the dimensions is not being wrapped. Example:
     *
     * gridDim = (6, 3, 1)
     * edgeRadii = (2, 3, 0)
     *
     * memory layout (numbers = x-coord, e = edgeCell
     *
     * eeeeeeeeee
     * eeeeeeeeee
     * eeeeeeeeee
     * ee012345ee
     * ee012345ee
     * ee012345ee
     * eeeeeeeeee
     * eeeeeeeeee
     * eeeeeeeeee
     */
    SetEdges(const Coord<3>& gridDim, const Coord<3>& edgeRadii, const CELL& cell) :
        gridDim(gridDim),
        edgeRadii(edgeRadii),
        cell(cell)
    {}

    template<int DIM_X, int DIM_Y, int DIM_Z, int INDEX>
    void operator()(
        const LibFlatArray::soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor,
        int *index)
    {
        Coord<3> expandedGridDim = gridDim + (edgeRadii * 2);
        Coord<3> dim;

        // west boundary:
        dim = Coord<3>(edgeRadii.x(), expandedGridDim.y(), expandedGridDim.z());
        setBox(CoordBox<3>(Coord<3>(), dim), accessor, index);
        // east boundary:
        setBox(CoordBox<3>(Coord<3>(gridDim.x() + edgeRadii.x()), dim), accessor, index);

        // top boundary:
        dim = Coord<3>(expandedGridDim.x(), edgeRadii.y(), expandedGridDim.z());
        setBox(CoordBox<3>(Coord<3>(), dim), accessor, index);
        // bottom boundary:
        setBox(CoordBox<3>(Coord<3>(0, gridDim.y() + edgeRadii.y()), dim), accessor, index);

        // south boundary:
        dim = Coord<3>(expandedGridDim.x(), expandedGridDim.y(), edgeRadii.z());
        setBox(CoordBox<3>(Coord<3>(), dim), accessor, index);
        // north boundary:
        setBox(CoordBox<3>(Coord<3>(0, 0, gridDim.z() + edgeRadii.z()), dim), accessor, index);
    }

private:
    Coord<3> gridDim;
    Coord<3> edgeRadii;
    CELL cell;

    template<int DIM_X, int DIM_Y, int DIM_Z, int INDEX>
    void setBox(
        const CoordBox<3>& box,
        const LibFlatArray::soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX>& accessor,
        int *index)
    {
        for (int z = box.origin.z(); z < (box.origin.z() + box.dimensions.z()); ++z) {
            for (int y = box.origin.y(); y < (box.origin.y() + box.dimensions.y()); ++y) {
                for (int x = box.origin.x(); x < (box.origin.x() + box.dimensions.x()); ++x) {
                    *index =
                        z * DIM_X * DIM_Y +
                        y * DIM_X +
                        x;

                    accessor << cell;
                }
            }
        }
    }
};

}

template<typename CELL,
         typename TOPOLOGY=Topologies::Cube<2>::Topology,
         bool TOPOLOGICALLY_CORRECT=false>
class SoAGrid : public GridBase<CELL, TOPOLOGY::DIM>
{
public:
    const static int DIM = TOPOLOGY::DIM;

    typedef CELL CellType;
    typedef TOPOLOGY Topology;
    typedef LibFlatArray::soa_grid<CELL> Delegate;

    explicit SoAGrid(
        const CoordBox<DIM>& box = CoordBox<DIM>(),
        const CELL &defaultCell = CELL(),
        const CELL &edgeCell = CELL(),
        const Coord<DIM>& topologicalDimensions = Coord<DIM>()) :
        edgeRadii(genEdgeRadii()),
        edgeCell(edgeCell),
        box(box),
        topoDimensions(topologicalDimensions)
    {
        actualDimensions = Coord<3>::diagonal(1);
        for (int i = 0; i < DIM; ++i) {
            actualDimensions[i] = box.dimensions[i];
        }

        delegate.resize(
            actualDimensions.x(),
            actualDimensions.y(),
            actualDimensions.z());
        fill(defaultCell);
        setEdge(edgeCell);
    }


    virtual void set(const Coord<DIM>& coord, const CELL& cell)
    {
        delegateSet(coord, cell);
    }

    virtual CELL get(const Coord<DIM>& coord) const
    {
        return getDelegate(coord);
    }

    virtual void setEdge(const CELL& cell)
    {
        edgeCell = cell;
        int index;

        Coord<3> dim = Coord<3>::diagonal(1);
        for (int i = 0; i < DIM; ++i) {
            dim[i] = box.dimensions[i];
        }
        delegate.callback(SoAGridHelpers::SetEdges<CELL>(dim, edgeRadii, cell), &index);
    }

    virtual const CELL& getEdge() const
    {
        return edgeCell;
    }

    virtual CoordBox<DIM> boundingBox() const
    {
        return box;
    }

    void fill(const CELL& cell)
    {
        // fixme
    }

private:
    Delegate delegate;
    Coord<3> edgeRadii;
    Coord<3> actualDimensions;
    CELL edgeCell;
    CoordBox<DIM> box;
    Coord<DIM> topoDimensions;

    static Coord<3> genEdgeRadii()
    {
        return Coord<3>(
            Topology::wrapsAxis(0) ? 1 : 0,
            Topology::wrapsAxis(1) ? 1 : 0,
            Topology::wrapsAxis(2) ? 1 : 0);
    }

    CELL delegateGet(const Coord<1>& coord) const
    {
        return delegate.get(coord.x(), 0, 0);
    }

    CELL delegateGet(const Coord<2>& coord) const
    {
        return delegate.get(coord.x(), coord.y(), 0);
    }

    CELL delegateGet(const Coord<3>& coord) const
    {
        return delegate.get(coord.x(), coord.y(), coord.z());
    }

    void delegateSet(const Coord<1>& coord, const CELL& cell)
    {
        return delegate.set(coord.x(), 0, 0, cell);
    }

    void delegateSet(const Coord<2>& coord, const CELL& cell)
    {
        return delegate.set(coord.x(), coord.y(), 0, cell);
    }

    void delegateSet(const Coord<3>& coord, const CELL& cell)
    {
        return delegate.set(coord.x(), coord.y(), coord.z(), cell);
    }
};

}

#endif
