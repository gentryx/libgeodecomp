#ifndef LIBGEODECOMP_MISC_SOAGRID_H
#define LIBGEODECOMP_MISC_SOAGRID_H

#include <libflatarray/flat_array.hpp>
#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/gridbase.h>
#include <libgeodecomp/misc/topologies.h>

namespace LibGeoDecomp {

namespace SoAGridHelpers {

template<typename CELL, bool INIT_INTERIOR>
class SetContent
{
public:
    /**
     * Helper class for initializing a SoAGrid. gridDim refers to the
     * size of the grid as requested by the user. edgeRadii specifies
     * how many layers of cells are to be padded around these,
     * representing the edgeCell. This procedure makes updating the
     * edgeCell slow, but eliminates all conditionals on the grid
     * boundaries. edgeRadii[D] will typically be 0 if the topology
     * wraps this dimension. It'll be equal to the stencil's radius if
     * the dimensions is not being wrapped. Example:
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
    SetContent(
        const Coord<3>& gridDim, const Coord<3>& edgeRadii, const CELL& edgeCell, const CELL& innerCell) :
        gridDim(gridDim),
        edgeRadii(edgeRadii),
        edgeCell(edgeCell),
        innerCell(innerCell)
    {}

    template<int DIM_X, int DIM_Y, int DIM_Z, int INDEX>
    void operator()(
        LibFlatArray::soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor,
        int *index)
    {
        for (int z = 0; z < gridDim.z(); ++z) {
            const CELL *cell1 = &innerCell;
            if ((z < edgeRadii.z()) || (z > (gridDim.z() - edgeRadii.z()))) {
                cell1 = &edgeCell;
            }

            for (int y = 0; y < gridDim.y(); ++y) {
                const CELL *cell2 = cell1;
                if ((y < edgeRadii.y()) || (y > (gridDim.z() - edgeRadii.y()))) {
                    cell2 = &edgeCell;
                }

                *index =
                    z * DIM_X * DIM_Y +
                    y * DIM_X;
                int x = 0;

                for (; x < edgeRadii.x(); ++x) {
                    accessor << edgeCell;
                    ++*index;
                }

                if (INIT_INTERIOR) {
                    for (; x < (gridDim.x() - edgeRadii.x()); ++x) {
                        accessor << *cell2;
                        ++*index;
                    }
                } else {
                    // we need to advance index and x manually, otherwise
                    // the following loop will erase the grid's interior:
                    int delta = gridDim.x() - 2 * edgeRadii.x();
                    x += delta;
                    *index += delta;
                }

                for (; x < gridDim.x(); ++x) {
                    accessor << edgeCell;
                    ++*index;
                }
            }

        }
    }

private:
    Coord<3> gridDim;
    Coord<3> edgeRadii;
    CELL edgeCell;
    CELL innerCell;
};

}

template<typename CELL,
         typename TOPOLOGY=Topologies::Cube<2>::Topology,
         bool TOPOLOGICALLY_CORRECT=false>
class SoAGrid : public GridBase<CELL, TOPOLOGY::DIM>
{
public:
    friend class SoAGridTest;

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
        actualDimensions += edgeRadii * 2;

        delegate.resize(
            actualDimensions.x(),
            actualDimensions.y(),
            actualDimensions.z());

        // init edges and interior
        int index;
        delegate.callback(SoAGridHelpers::SetContent<CELL, true>(
                              actualDimensions, edgeRadii, edgeCell, defaultCell), &index);

    }


    virtual void set(const Coord<DIM>& absoluteCoord, const CELL& cell)
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

    virtual CELL get(const Coord<DIM>& absoluteCoord) const
    {
        std::cout << "get(" << absoluteCoord << ", " << edgeRadii << "\n";
        Coord<DIM> relativeCoord = absoluteCoord - box.origin;
        std::cout << "relativeCoord: " << relativeCoord << "\n";
        if (TOPOLOGICALLY_CORRECT) {
            relativeCoord = Topology::normalize(relativeCoord, topoDimensions);
        }
        std::cout << "relativeCoord: " << relativeCoord << "\n";
        if (Topology::isOutOfBounds(relativeCoord, box.dimensions)) {
            std::cout << "isOutOfBounds\n";
            return edgeCell;
        }
        std::cout << "relativeCoord: " << relativeCoord << "\n";
        return delegateGet(relativeCoord);
    }

    virtual void setEdge(const CELL& cell)
    {
        edgeCell = cell;
        int index;

        CELL dummy;
        delegate.callback(SoAGridHelpers::SetContent<CELL, false>(
                              actualDimensions, edgeRadii, edgeCell, dummy), &index);
    }

    virtual const CELL& getEdge() const
    {
        return edgeCell;
    }

    virtual CoordBox<DIM> boundingBox() const
    {
        return box;
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
            Topology::wrapsAxis(0) || (Topology::DIM < 1) ? 0 : 1,
            Topology::wrapsAxis(1) || (Topology::DIM < 2) ? 0 : 1,
            Topology::wrapsAxis(2) || (Topology::DIM < 3) ? 0 : 1);
    }

    CELL delegateGet(const Coord<1>& coord) const
    {
        return delegate.get(coord.x() + edgeRadii.x(), edgeRadii.y(), edgeRadii.z());
    }

    CELL delegateGet(const Coord<2>& coord) const
    {
        return delegate.get(coord.x() + edgeRadii.x(), coord.y() + edgeRadii.y(), edgeRadii.z());
    }

    CELL delegateGet(const Coord<3>& coord) const
    {
        return delegate.get(coord.x() + edgeRadii.x(), coord.y() + edgeRadii.y(), coord.z() + edgeRadii.z());
    }

    void delegateSet(const Coord<1>& coord, const CELL& cell)
    {
        return delegate.set(coord.x() + edgeRadii.x(),  edgeRadii.y(), edgeRadii.z(), cell);
    }

    void delegateSet(const Coord<2>& coord, const CELL& cell)
    {
        return delegate.set(coord.x() + edgeRadii.x(), coord.y() + edgeRadii.y(), edgeRadii.z(), cell);
    }

    void delegateSet(const Coord<3>& coord, const CELL& cell)
    {
        return delegate.set(coord.x() + edgeRadii.x(), coord.y() + edgeRadii.y(), coord.z() + edgeRadii.z(), cell);
    }
};

}

#endif
