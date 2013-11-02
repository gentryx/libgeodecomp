#ifndef LIBGEODECOMP_STORAGE_SOAGRID_H
#define LIBGEODECOMP_STORAGE_SOAGRID_H

#include <libflatarray/flat_array.hpp>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/topologies.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/storage/gridbase.h>

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
            bool onEdge1 = false;
            const CELL *cell1 = &innerCell;
            if ((z < edgeRadii.z()) || (z >= (gridDim.z() - edgeRadii.z()))) {
                cell1 = &edgeCell;
                onEdge1 = true;
            }

            for (int y = 0; y < gridDim.y(); ++y) {
                bool onEdge2 = onEdge1;
                const CELL *cell2 = cell1;
                if ((y < edgeRadii.y()) || (y >= (gridDim.y() - edgeRadii.y()))) {
                    cell2 = &edgeCell;
                    onEdge2 = true;
                }

                *index =
                    z * DIM_X * DIM_Y +
                    y * DIM_X;
                int x = 0;

                for (; x < edgeRadii.x(); ++x) {
                    accessor << edgeCell;
                    ++*index;
                }

                if (onEdge2 || INIT_INTERIOR) {
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
         typename TOPOLOGY = Topologies::Cube<2>::Topology,
         bool TOPOLOGICALLY_CORRECT = false>
class SoAGrid : public GridBase<CELL, TOPOLOGY::DIM>
{
public:
    friend class SoAGridTest;

    const static int DIM = TOPOLOGY::DIM;

    typedef CELL CellType;
    typedef TOPOLOGY Topology;
    typedef LibFlatArray::soa_grid<CELL> Delegate;
    typedef typename APITraits::SelectStencil<CELL>::Value Stencil;

    explicit SoAGrid(
        const CoordBox<DIM>& box = CoordBox<DIM>(),
        const CELL &defaultCell = CELL(),
        const CELL &edgeCell = CELL(),
        const Coord<DIM>& topologicalDimensions = Coord<DIM>()) :
        edgeRadii(calcEdgeRadii()),
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

    virtual void set(const Streak<DIM>& streak, const CELL *cells)
    {
        Coord<DIM> relativeCoord = streak.origin - box.origin;
        if (TOPOLOGICALLY_CORRECT) {
            relativeCoord = Topology::normalize(relativeCoord, topoDimensions);
        }

        delegateSet(relativeCoord, cells, streak.length());
    }

    virtual CELL get(const Coord<DIM>& absoluteCoord) const
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

    virtual void get(const Streak<DIM>& streak, CELL *cells) const
    {
        Coord<DIM> relativeCoord = streak.origin - box.origin;
        if (TOPOLOGICALLY_CORRECT) {
            relativeCoord = Topology::normalize(relativeCoord, topoDimensions);
        }

        delegateGet(relativeCoord, cells, streak.length());
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

    const Coord<3>& getEdgeRadii() const
    {
        return edgeRadii;
    }

    virtual CoordBox<DIM> boundingBox() const
    {
        return box;
    }

    template<typename FUNCTOR>
    void callback(FUNCTOR functor) const
    {
        int index = 0;
        delegate.callback(functor, &index);
    }

    template<typename FUNCTOR>
    void callback(SoAGrid<CELL, TOPOLOGY, TOPOLOGICALLY_CORRECT> *newGrid, FUNCTOR functor) const
    {
        delegate.callback(&newGrid->delegate, functor);
    }

    void saveRegion(char *target, const Region<DIM>& region)
    {
        char *dataIterator = target;

        for (typename Region<DIM>::StreakIterator i = region.beginStreak();
             i != region.endStreak();
             ++i) {
            Streak<DIM> s = *i;
            size_t length = s.length();
            int x = s.origin.x() + edgeRadii.x() - box.origin.x();
            int y = s.origin.y() + edgeRadii.y() - box.origin.y();
            int z = s.origin.z() + edgeRadii.z() - box.origin.z();
            delegate.save(x, y, z, dataIterator, length);
            dataIterator += length * sizeof(CELL);
        }

    }

    void loadRegion(char *source, const Region<DIM>& region)
    {
        char *dataIterator = source;

        for (typename Region<DIM>::StreakIterator i = region.beginStreak();
             i != region.endStreak();
             ++i) {
            Streak<DIM> s = *i;
            size_t length = s.length();
            int x = s.origin.x() + edgeRadii.x() - box.origin.x();
            int y = s.origin.y() + edgeRadii.y() - box.origin.y();
            int z = s.origin.z() + edgeRadii.z() - box.origin.z();
            delegate.load(x, y, z, dataIterator, length);
            dataIterator += length * sizeof(CELL);
        }

    }

private:
    Delegate delegate;
    Coord<3> edgeRadii;
    Coord<3> actualDimensions;
    CELL edgeCell;
    CoordBox<DIM> box;
    Coord<DIM> topoDimensions;

    static Coord<3> calcEdgeRadii()
    {
        return Coord<3>(
            Topology::wrapsAxis(0) || (Topology::DIM < 1) ? 0 : Stencil::RADIUS,
            Topology::wrapsAxis(1) || (Topology::DIM < 2) ? 0 : Stencil::RADIUS,
            Topology::wrapsAxis(2) || (Topology::DIM < 3) ? 0 : Stencil::RADIUS);
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
        return delegate.set(
            edgeRadii.x() + coord.x(),
            edgeRadii.y(),
            edgeRadii.z(),
            cell);
    }

    void delegateSet(const Coord<2>& coord, const CELL& cell)
    {
        return delegate.set(
            edgeRadii.x() + coord.x(),
            edgeRadii.y() + coord.y(),
            edgeRadii.z(),
            cell);
    }

    void delegateSet(const Coord<3>& coord, const CELL& cell)
    {
        return delegate.set(
            edgeRadii.x() + coord.x(),
            edgeRadii.y() + coord.y(),
            edgeRadii.z() + coord.z(),
            cell);
    }

    void delegateSet(const Coord<1>& coord, const CELL *cells, int count)
    {
        return delegate.set(
            edgeRadii.x() + coord.x(),
            edgeRadii.y(),
            edgeRadii.z(),
            cells,
            count);
    }

    void delegateSet(const Coord<2>& coord, const CELL *cells, int count)
    {
        return delegate.set(
            edgeRadii.x() + coord.x(),
            edgeRadii.y() + coord.y(),
            edgeRadii.z(),
            cells,
            count);
    }

    void delegateSet(const Coord<3>& coord, const CELL *cells, int count)
    {
        return delegate.set(
            edgeRadii.x() + coord.x(),
            edgeRadii.y() + coord.y(),
            edgeRadii.z() + coord.z(),
            cells,
            count);
    }
};

}

#endif
