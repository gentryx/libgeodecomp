#ifndef LIBGEODECOMP_STORAGE_SOAGRID_H
#define LIBGEODECOMP_STORAGE_SOAGRID_H

#include <libflatarray/flat_array.hpp>

#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/geometry/topologies.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/stringops.h>
#include <libgeodecomp/storage/gridbase.h>
#include <libgeodecomp/storage/selector.h>

namespace LibGeoDecomp {

namespace SoAGridHelpers {

/**
 * See below:
 */
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
     * gridDim = (10, 9, 1)
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

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(LibFlatArray::soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor) const
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

                accessor.index =
                    z * DIM_X * DIM_Y +
                    y * DIM_X;
                int x = 0;

                for (; x < edgeRadii.x(); ++x) {
                    accessor << edgeCell;
                    ++accessor.index;
                }

                if (onEdge2 || INIT_INTERIOR) {
                    for (; x < (gridDim.x() - edgeRadii.x()); ++x) {
                        accessor << *cell2;
                        ++accessor.index;
                    }
                } else {
                    // we need to advance index and x manually, otherwise
                    // the following loop will erase the grid's interior:
                    int delta = gridDim.x() - 2 * edgeRadii.x();
                    x += delta;
                    accessor.index += delta;
                }

                for (; x < gridDim.x(); ++x) {
                    accessor << edgeCell;
                    ++accessor.index;
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

/**
 * A simple functor for wrapping index calculation within the SoA
 * layout. It's purpose is to hide differences in the calculation when
 * using 1D, 2D or 3D coords. LibFlatArray internally always uses a 3D
 * layout, thus our edgeRadii are also always 3D.
 */
template<int DIM_X, int DIM_Y, int DIM_Z>
class GenIndex
{
public:
    int operator()(const Coord<1>& coord, const Coord<3>& edgeRadii) const
    {
        return
            coord.x() + edgeRadii.x() +
            DIM_X * edgeRadii.y() +
            DIM_X * DIM_Y * edgeRadii.z();
    }

    int operator()(const Coord<2>& coord, const Coord<3>& edgeRadii) const
    {
        return
            coord.x() + edgeRadii.x() +
            DIM_X * (coord.y() + edgeRadii.y()) +
            DIM_X * DIM_Y * edgeRadii.z();
    }

    int operator()(const Coord<3>& coord, const Coord<3>& edgeRadii) const
    {
        return
            coord.x() + edgeRadii.x() +
            DIM_X * (coord.y() + edgeRadii.y()) +
            DIM_X * DIM_Y * (coord.z() + edgeRadii.z());
    }
};

/**
 * Extract a single member variable from a SoA grid
 */
template<typename CELL, int DIM>
class SaveMember
{
public:
    SaveMember(
        char *target,
        MemoryLocation::Location targetLocation,
        const Selector<CELL>& selector,
        const Region<DIM>& region,
        const Coord<DIM>& origin,
        const Coord<3>& edgeRadii) :
        target(target),
        targetLocation(targetLocation),
        selector(selector),
        region(region),
        origin(origin),
        edgeRadii(edgeRadii)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(LibFlatArray::soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor) const
    {
        char *currentTarget = target;

        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            accessor.index = GenIndex<DIM_X, DIM_Y, DIM_Z>()(i->origin - origin, edgeRadii);
            const char *data = accessor.access_member(selector.sizeOfMember(), selector.offset());
            selector.copyStreakOut(
                data,
                MemoryLocation::HOST,
                currentTarget,
                targetLocation,
                i->length(),
                DIM_X * DIM_Y * DIM_Z);
            currentTarget += selector.sizeOfExternal() * i->length();
        }
    }

private:
    char *target;
    MemoryLocation::Location targetLocation;
    const Selector<CELL>& selector;
    const Region<DIM>& region;
    const Coord<DIM>& origin;
    const Coord<3>& edgeRadii;
    long memberOffset;
};

/**
 * Counterpart to SaveMember
 */
template<typename CELL, int DIM>
class LoadMember
{
public:
    LoadMember(
        const char *source,
        MemoryLocation::Location sourceLocation,
        const Selector<CELL>& selector,
        const Region<DIM>& region,
        const Coord<DIM>& origin,
        const Coord<3>& edgeRadii) :
        source(source),
        sourceLocation(sourceLocation),
        selector(selector),
        region(region),
        origin(origin),
        edgeRadii(edgeRadii)
    {}

    template<long DIM_X, long DIM_Y, long DIM_Z, long INDEX>
    void operator()(LibFlatArray::soa_accessor<CELL, DIM_X, DIM_Y, DIM_Z, INDEX> accessor) const
    {
        const char *currentSource = source;

        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            accessor.index = GenIndex<DIM_X, DIM_Y, DIM_Z>()(i->origin - origin, edgeRadii);

            char *currentTarget = accessor.access_member(selector.sizeOfMember(), selector.offset());
            selector.copyStreakIn(
                currentSource,
                sourceLocation,
                currentTarget,
                MemoryLocation::HOST,
                i->length(),
                DIM_X * DIM_Y * DIM_Z);

            currentSource += selector.sizeOfExternal() * i->length();
        }
    }

private:
    const char *source;
    MemoryLocation::Location sourceLocation;
    const Selector<CELL>& selector;
    const Region<DIM>& region;
    const Coord<DIM>& origin;
    const Coord<3>& edgeRadii;
    long memberOffset;
};

template<typename ITERATOR, int DIM>
class OffsetStreakIterator
{
public:
    inline OffsetStreakIterator(const ITERATOR& delegate, Coord<3> offset) :
        delegate(delegate),
        offset(offset)
    {
        reset();
    }

    inline bool operator==(const OffsetStreakIterator other)
    {
        return delegate == other.delegate;
    }

    inline bool operator!=(const OffsetStreakIterator other)
    {
        return delegate != other.delegate;
    }

    inline const OffsetStreakIterator& operator*() const
    {
        return *this;
    }

    inline const OffsetStreakIterator *operator->() const
    {
        return this;
    }

    inline OffsetStreakIterator& operator++()
    {
        ++delegate;
        reset();

        return *this;
    }

    inline int length() const
    {
        return delegate->length();
    }

    Coord<3> origin;

private:
    ITERATOR delegate;
    Coord<3> offset;

    inline void reset()
    {
        origin = offset;

        for (int i = 0; i < DIM; ++i) {
            origin[i] += delegate->origin[i];
        }
    }
};

}

/**
 * Grid class which corresponds to DisplacedGrid, but utilizes an
 * "Struct of Arrays" (SoA) memory layout. This is beneficial for
 * vectorization.
 */
template<typename CELL,
         typename TOPOLOGY = Topologies::Cube<2>::Topology,
         bool TOPOLOGICALLY_CORRECT = false>
class SoAGrid : public GridBase<CELL, TOPOLOGY::DIM>
{
public:
    friend class SoAGridTest;
    friend class SelectorTest;

    using typename GridBase<CELL, TOPOLOGY::DIM>::BufferType;
    using GridBase<CELL, TOPOLOGY::DIM>::topoDimensions;
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

    explicit SoAGrid(
        const CoordBox<DIM>& box = CoordBox<DIM>(),
        const CELL& defaultCell = CELL(),
        const CELL& edgeCell = CELL(),
        const Coord<DIM>& topologicalDimensions = Coord<DIM>()) :
        GridBase<CELL, TOPOLOGY::DIM>(topologicalDimensions),
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

        // init edges and interior
        delegate.callback(
            SoAGridHelpers::SetContent<CELL, true>(
                actualDimensions, edgeRadii, edgeCell, defaultCell));
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
        CELL dummy;

        delegate.callback(
            SoAGridHelpers::SetContent<CELL, false>(
                actualDimensions, edgeRadii, edgeCell, dummy));
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
        delegate.callback(functor);
    }

    template<typename FUNCTOR>
    void callback(SoAGrid<CELL, TOPOLOGY, TOPOLOGICALLY_CORRECT> *newGrid, FUNCTOR functor) const
    {
        delegate.callback(&newGrid->delegate, functor);
    }

    void saveRegion(BufferType *target, const Region<DIM>& region, const Coord<DIM>& offset = Coord<DIM>()) const
    {
        SerializationBuffer<CELL>::resize(target, region);
        Coord<3> actualOffset = edgeRadii;
        for (int i = 0; i < DIM; ++i) {
            actualOffset[i] += -box.origin[i] + offset[i];
        }

        typedef SoAGridHelpers::OffsetStreakIterator<typename Region<DIM>::StreakIterator, DIM> StreakIteratorType;
        StreakIteratorType start(region.beginStreak(), actualOffset);
        StreakIteratorType end(  region.endStreak(),   actualOffset);

        delegate.save(start, end, target->data(), region.size());
    }

    void loadRegion(const BufferType& source, const Region<DIM>& region, const Coord<DIM>& offset = Coord<DIM>())
    {
        std::size_t expectedMinimumSize = SerializationBuffer<CELL>::storageSize(region);
        if (source.size() < expectedMinimumSize) {
            throw std::logic_error(
                "source buffer too small (is " + StringOps::itoa(source.size()) +
                ", expected at least: " + StringOps::itoa(expectedMinimumSize) + ")");
        }

        Coord<3> actualOffset = edgeRadii;
        for (int i = 0; i < DIM; ++i) {
            actualOffset[i] += -box.origin[i] + offset[i];
        }

        typedef SoAGridHelpers::OffsetStreakIterator<typename Region<DIM>::StreakIterator, DIM> StreakIteratorType;
        StreakIteratorType start(region.beginStreak(), actualOffset);
        StreakIteratorType end(  region.endStreak(),   actualOffset);

        delegate.load(start, end, source.data(), region.size());
    }

    static Coord<3> calcEdgeRadii()
    {
        return Coord<3>(
            Topology::wrapsAxis(0) || (Topology::DIM < 1) ? 0 : Stencil::RADIUS,
            Topology::wrapsAxis(1) || (Topology::DIM < 2) ? 0 : Stencil::RADIUS,
            Topology::wrapsAxis(2) || (Topology::DIM < 3) ? 0 : Stencil::RADIUS);
    }

protected:
    void saveMemberImplementation(
        char *target,
        MemoryLocation::Location targetLocation,
        const Selector<CELL>& selector,
        const Region<DIM>& region) const
    {
        delegate.callback(
            SoAGridHelpers::SaveMember<CELL, DIM>(
                target, targetLocation, selector, region, box.origin, edgeRadii));
    }

    void loadMemberImplementation(
        const char *source,
        MemoryLocation::Location sourceLocation,
        const Selector<CELL>& selector,
        const Region<DIM>& region)
    {
        delegate.callback(
            SoAGridHelpers::LoadMember<CELL, DIM>(
                source, sourceLocation, selector, region, box.origin, edgeRadii));
    }

private:
    Delegate delegate;
    Coord<3> edgeRadii;
    Coord<3> actualDimensions;
    CELL edgeCell;
    CoordBox<DIM> box;

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
