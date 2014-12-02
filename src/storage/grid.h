#ifndef LIBGEODECOMP_STORAGE_GRID_H
#define LIBGEODECOMP_STORAGE_GRID_H

#include <libflatarray/aligned_allocator.hpp>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/geometry/topologies.h>
#include <libgeodecomp/storage/coordmap.h>
#include <libgeodecomp/storage/gridbase.h>
#include <libgeodecomp/storage/selector.h>

// CodeGear's C++ compiler isn't compatible with boost::multi_array
// (at least the version that ships with C++ Builder 2009)
#ifndef __CODEGEARC__
#include <boost/multi_array.hpp>
#else
#include <libgeodecomp/misc/supervector.h>
#endif

#include <boost/foreach.hpp>
#include <iostream>

namespace LibGeoDecomp {

template<typename CELL_TYPE, typename GRID_TYPE>
class CoordMap;

namespace GridHelpers {

template<int DIM>
class FillCoordBox;

template<>
class FillCoordBox<1>
{
public:
    template<typename GRID, typename CELL>
    void operator()(const Coord<1>& origin, const Coord<1>& dim, GRID *grid, const CELL& cell)
    {
        CELL *cursor = &(*grid)[origin];
        std::fill(cursor, cursor + dim.x(), cell);
    }
};

template<>
class FillCoordBox<2>
{
public:
    template<typename GRID, typename CELL>
    void operator()(const Coord<2>& origin, const Coord<2>& dim, GRID *grid, const CELL& cell)
    {
        int maxY = origin.y() + dim.y();
        Coord<2> c = origin;
        for (; c.y() < maxY; ++c.y()) {
            CELL *cursor = &(*grid)[c];
            std::fill(cursor, cursor + dim.x(), cell);
        }
    }
};

template<>
class FillCoordBox<3>
{
public:
    template<typename GRID, typename CELL>
    void operator()(const Coord<3>& origin, const Coord<3>& dim, GRID *grid, const CELL& cell)
    {
        int maxY = origin.y() + dim.y();
        int maxZ = origin.z() + dim.z();
        Coord<3> c = origin;

        for (; c.z() < maxZ; ++c.z()) {
            for (c.y() = origin.y(); c.y() < maxY; ++c.y()) {
                CELL *cursor = &(*grid)[c];
                std::fill(cursor, cursor + dim.x(), cell);
            }
        }
    }
};

}

/**
 * A multi-dimensional regular grid
 */
template<typename CELL_TYPE, typename TOPOLOGY=Topologies::Cube<2>::Topology>
class Grid : public GridBase<CELL_TYPE, TOPOLOGY::DIM>
{
public:
    friend class GridTest;
    friend class ParallelStripingSimulatorTest;
    const static int DIM = TOPOLOGY::DIM;

#ifndef __CODEGEARC__
    typedef typename boost::detail::multi_array::sub_array<CELL_TYPE, DIM - 1> SliceRef;
    typedef typename boost::detail::multi_array::const_sub_array<CELL_TYPE, 1> ConstSliceRef;
    typedef typename boost::multi_array<
        // always align on cache line boundaries
        CELL_TYPE, DIM, LibFlatArray::aligned_allocator<CELL_TYPE, 64> > CellMatrix;
    typedef typename CellMatrix::index Index;
#else
    typedef std::vector<CELL_TYPE>& SliceRef;
    typedef const std::vector<CELL_TYPE>& ConstSliceRef;
    typedef int Index;
#endif

    typedef TOPOLOGY Topology;
    typedef CELL_TYPE Cell;
    typedef CoordMap<CELL_TYPE, Grid<CELL_TYPE, TOPOLOGY> > CoordMapType;

    explicit Grid(
        const Coord<DIM>& dim = Coord<DIM>(),
        const CELL_TYPE& defaultCell = CELL_TYPE(),
        const CELL_TYPE& edgeCell = CELL_TYPE()) :
        dimensions(dim),
        cellMatrix(dim.toExtents()),
        edgeCell(edgeCell)
    {
        CoordBox<DIM> box(Coord<DIM>(), dim);
        for (typename CoordBox<DIM>::StreakIterator i = box.beginStreak(); i != box.endStreak(); ++i) {
            Streak<DIM> s = *i;
            CELL_TYPE *start = &(*this)[s.origin];
            CELL_TYPE *end   = start + s.length();
            std::fill(start, end, defaultCell);
        }
    }

    explicit Grid(const GridBase<CELL_TYPE, DIM>& base) :
        dimensions(base.dimensions()),
        cellMatrix(base.dimensions().toExtents()),
        edgeCell(base.getEdge())
    {
        CoordBox<DIM> box = base.boundingBox();
        for (typename CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
            set(*i - box.origin, base.get(*i));
        }
    }

    Grid& operator=(const Grid& other)
    {
        resize(other.getDimensions());
        std::copy(other.cellMatrix.begin(), other.cellMatrix.end(), cellMatrix.begin());
        edgeCell = other.edgeCell;

        return *this;
    }

    inline void resize(const Coord<DIM>& newDim)
    {
        // temporarly resize to 0-sized array to avoid having two
        // large arrays simultaneously allocated. somehow I feel
        // boost::multi_array::resize should be responsible for
        // this...
        Coord<DIM> tempDim;
        cellMatrix.resize(tempDim.toExtents());
        dimensions = newDim;
        cellMatrix.resize(newDim.toExtents());
    }

    /**
     * returns a map that is referenced by relative coordinates from the
     * originating coordinate coord.
     */
    inline CoordMapType getNeighborhood(const Coord<DIM>& center) const
    {
        return CoordMapType(center, this);
    }

    inline CELL_TYPE& getEdgeCell()
    {
        return edgeCell;
    }

    inline const CELL_TYPE& getEdgeCell() const
    {
        return edgeCell;
    }

    inline CELL_TYPE *baseAddress()
    {
        return &(*this)[Coord<DIM>()];
    }

    inline const CELL_TYPE *baseAddress() const
    {
        return &(*this)[Coord<DIM>()];
    }

    inline CELL_TYPE& operator[](const Coord<DIM>& coord)
    {
        return Topology::locate(*this, coord);
    }

    inline const CELL_TYPE& operator[](const Coord<DIM>& coord) const
    {
        return (const_cast<Grid&>(*this))[coord];
    }

    /**
     * WARNING: this operator doesn't honor topology properties
     */
    inline const ConstSliceRef operator[](const Index y) const
    {
        return cellMatrix[y];
    }

    /**
     * WARNING: this operator doesn't honor topology properties
     */
    inline SliceRef operator[](const Index y)
    {
        return cellMatrix[y];
    }

    inline bool operator==(const Grid& other) const
    {
        if (boundingBox() == CoordBox<DIM>() &&
            other.boundingBox() == CoordBox<DIM>()) {
            return true;
        }

        return
            (edgeCell   == other.edgeCell) &&
            (cellMatrix == other.cellMatrix);
    }

    inline bool operator==(const GridBase<CELL_TYPE, TOPOLOGY::DIM>& other) const
    {
        if (boundingBox() != other.boundingBox()) {
            return false;
        }

        if (edgeCell != other.getEdge()) {
            return false;
        }

        CoordBox<DIM> box = boundingBox();
        for (typename CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
            if ((*this)[*i] != other.get(*i)) {
                return false;
            }
        }

        return true;
    }

    inline bool operator!=(const Grid& other) const
    {
        return !(*this == other);
    }

    inline bool operator!=(const GridBase<CELL_TYPE, TOPOLOGY::DIM>& other) const
    {
        return !(*this == other);
    }

    virtual void set(const Coord<DIM>& coord, const CELL_TYPE& cell)
    {
        (*this)[coord] = cell;
    }

    virtual void set(const Streak<DIM>& streak, const CELL_TYPE *cells)
    {
        Coord<DIM> cursor = streak.origin;
        for (; cursor.x() < streak.endX; ++cursor.x()) {
            (*this)[cursor] = *cells;
            ++cells;
        }
    }

    virtual CELL_TYPE get(const Coord<DIM>& coord) const
    {
        return (*this)[coord];
    }

    virtual void get(const Streak<DIM>& streak, CELL_TYPE *cells) const
    {
        Coord<DIM> cursor = streak.origin;
        for (; cursor.x() < streak.endX; ++cursor.x()) {
            *cells = (*this)[cursor];
            ++cells;
        }
    }

    virtual void setEdge(const CELL_TYPE& cell)
    {
        getEdgeCell() = cell;
    }

    virtual const CELL_TYPE& getEdge() const
    {
        return getEdgeCell();
    }

    void fill(const CoordBox<DIM>& box, const CELL_TYPE& cell)
    {
        GridHelpers::FillCoordBox<DIM>()(box.origin, box.dimensions, this, cell);
    }

    inline const Coord<DIM>& getDimensions() const
    {
        return dimensions;
    }

    inline std::string toString() const
    {
        std::ostringstream message;
        message << "Grid<" << DIM << ">(\n"
                << "boundingBox: " << boundingBox()  << "\n"
                << "edgeCell:\n"
                << edgeCell << "\n";

        CoordBox<DIM> box = boundingBox();
        for (typename CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
            message << "Coord" << *i << ":\n" << (*this)[*i] << "\n";
        }

        message << ")";
        return message.str();
    }

    virtual CoordBox<DIM> boundingBox() const
    {
        return CoordBox<DIM>(Coord<DIM>(), dimensions);
    }

protected:
    void saveMemberImplementation(
        char *target, const Selector<CELL_TYPE>& selector, const Region<DIM>& region) const
    {
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            selector.copyMemberOut(&(*this)[i->origin], target, i->length());
            target += selector.sizeOfExternal() * i->length();
        }
    }

    void loadMemberImplementation(
        const char *source, const Selector<CELL_TYPE>& selector, const Region<DIM>& region)
    {
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            selector.copyMemberIn(source, &(*this)[i->origin], i->length());
            source += selector.sizeOfExternal() * i->length();
        }
    }

private:
    Coord<DIM> dimensions;
    CellMatrix cellMatrix;
    CELL_TYPE edgeCell;
};

template<typename _CharT, typename _Traits, typename _CellT, typename _TopologyT>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const Grid<_CellT, _TopologyT>& grid)
{
    __os << grid.toString();
    return __os;
}

}

#endif
