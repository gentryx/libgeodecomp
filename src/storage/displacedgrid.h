#ifndef LIBGEODECOMP_STORAGE_DISPLACEDGRID_H
#define LIBGEODECOMP_STORAGE_DISPLACEDGRID_H

#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/storage/grid.h>

namespace LibGeoDecomp {

/**
 * A grid whose origin has been shiftet by a certain offset. If
 * TOPOLOGICALLY_CORRECT is set to true, the coordinates will be
 * normalized according to the given topology and some superordinate
 * dimension (see topologicalDimensions()) before using them for
 * access. Useful for writing topology agnostic code that should work
 * an a torus, too.
 */
template<typename CELL_TYPE,
         typename TOPOLOGY=Topologies::Cube<2>::Topology,
         bool TOPOLOGICALLY_CORRECT=false>
class DisplacedGrid : public GridBase<CELL_TYPE, TOPOLOGY::DIM>
{
public:
    const static int DIM = TOPOLOGY::DIM;

    typedef CELL_TYPE Cell;
    typedef TOPOLOGY Topology;
    typedef typename boost::multi_array<CELL_TYPE, DIM>::index Index;
    typedef Grid<CELL_TYPE, TOPOLOGY> Delegate;
    typedef CoordMap<CELL_TYPE, Delegate> CoordMapType;

    explicit DisplacedGrid(
        const CoordBox<DIM>& box = CoordBox<DIM>(),
        const CELL_TYPE& defaultCell = CELL_TYPE(),
        const CELL_TYPE& edgeCell = CELL_TYPE(),
        const Coord<DIM>& topologicalDimensions = Coord<DIM>()) :
        delegate(box.dimensions, defaultCell, edgeCell),
        origin(box.origin),
        topoDimensions(topologicalDimensions)
    {}

    explicit DisplacedGrid(
        const Delegate& grid,
        const Coord<DIM>& origin=Coord<DIM>()) :
        delegate(grid),
        origin(origin)
    {}

    inline const Coord<DIM>& topologicalDimensions() const
    {
        return topoDimensions;
    }

    inline Coord<DIM>& topologicalDimensions()
    {
        return topoDimensions;
    }

    inline CELL_TYPE *baseAddress()
    {
        return delegate.baseAddress();
    }

    inline const CELL_TYPE *baseAddress() const
    {
        return delegate.baseAddress();
    }

    inline const Coord<DIM>& getOrigin() const
    {
        return origin;
    }

    inline const CELL_TYPE& getEdgeCell() const
    {
        return delegate.getEdgeCell();
    }

    inline CELL_TYPE& getEdgeCell()
    {
        return delegate.getEdgeCell();
    }

    inline void setOrigin(const Coord<DIM>& newOrigin)
    {
        origin = newOrigin;
    }

    inline void resize(const CoordBox<DIM>& box)
    {
        delegate.resize(box.dimensions);
        origin = box.origin;
    }

    inline CELL_TYPE& operator[](const Coord<DIM>& absoluteCoord)
    {
        Coord<DIM> relativeCoord = absoluteCoord - origin;
        if (TOPOLOGICALLY_CORRECT) {
            relativeCoord = Topology::normalize(relativeCoord, topoDimensions);
        }
        return delegate[relativeCoord];
    }

    inline const CELL_TYPE& operator[](const Coord<DIM>& absoluteCoord) const
    {
        return (const_cast<DisplacedGrid&>(*this))[absoluteCoord];
    }

    virtual void set(const Coord<DIM>& coord, const CELL_TYPE& cell)
    {
        (*this)[coord] = cell;
    }

    virtual void set(const Streak<DIM>& streak, const CELL_TYPE *cells)
    {
	delegate.set(Streak<DIM>(streak.origin - origin,
				 streak.endX - origin.x()),
		     cells);
    }

    virtual CELL_TYPE get(const Coord<DIM>& coord) const
    {
        return (*this)[coord];
    }

    virtual void get(const Streak<DIM>& streak, CELL_TYPE *cells) const
    {
	delegate.get(Streak<DIM>(streak.origin - origin,
				 streak.endX - origin.x()),
		     cells);
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
        delegate.fill(CoordBox<DIM>(box.origin - origin, box.dimensions), cell);
    }

    inline void paste(const GridBase<CELL_TYPE, DIM>& grid, const Region<DIM>& region)
    {
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            grid.get(*i, &(*this)[i->origin]);
        }
    }

    inline const Coord<DIM>& getDimensions() const
    {
        return delegate.getDimensions();
    }

    virtual CoordBox<DIM> boundingBox() const
    {
        return CoordBox<DIM>(origin, delegate.getDimensions());
    }

    inline CoordMapType getNeighborhood(const Coord<DIM>& center) const
    {
        Coord<DIM> relativeCoord = center - origin;
        if (TOPOLOGICALLY_CORRECT) {
            relativeCoord = Topology::normalize(relativeCoord, topoDimensions);
        }
        return CoordMapType(relativeCoord, &delegate);
    }

    inline const Delegate *vanillaGrid() const
    {
        return &delegate;
    }

    inline Delegate *vanillaGrid()
    {
        return &delegate;
    }

    inline std::string toString() const
    {
        std::ostringstream message;
        message << "DisplacedGrid<" << DIM << ">(\n"
                << "  origin: " << origin << "\n"
                << "  delegate:\n"
                << delegate
                << ")";
        return message.str();
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
    Delegate delegate;
    Coord<DIM> origin;
    Coord<DIM> topoDimensions;
};

}

template<typename _CharT, typename _Traits, typename _CellT, typename _Topology, bool _Correctness>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const LibGeoDecomp::DisplacedGrid<_CellT, _Topology, _Correctness>& grid)
{
    __os << grid.toString();
    return __os;
}

#endif
