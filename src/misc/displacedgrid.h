#ifndef _libgeodecomp_misc_displacedgrid_h_
#define _libgeodecomp_misc_displacedgrid_h_

#include <libgeodecomp/misc/coordbox.h>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/misc/region.h>

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
class DisplacedGrid : public GridBase<CELL_TYPE, TOPOLOGY::DIMENSIONS>
{
public:
    const static int DIM = TOPOLOGY::DIMENSIONS;

    typedef CELL_TYPE CellType;
    typedef TOPOLOGY Topology;
    typedef typename boost::multi_array<CELL_TYPE, DIM>::index Index;
    typedef Grid<CELL_TYPE, TOPOLOGY> Delegate;
    typedef CoordMap<CELL_TYPE, Delegate> MyCoordMap;

    explicit DisplacedGrid(
        const CoordBox<DIM>& box = CoordBox<DIM>(),
        const CELL_TYPE &_defaultCell=CELL_TYPE(),
        const Coord<DIM>& topologicalDimensions=Coord<DIM>()) :
        delegate(box.dimensions, _defaultCell),
        origin(box.origin),
        topoDimensions(topologicalDimensions)
    { }


    DisplacedGrid(const Delegate& _grid,
                  const Coord<DIM>& _origin=Coord<DIM>()) :
        delegate(_grid),
        origin(_origin)
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

    inline void resize(const CoordBox<DIM>& box)
    {
        delegate.resize(box.dimensions);
        origin = box.origin;
    }

    inline CELL_TYPE& operator[](const Coord<DIM>& absoluteCoord)
    {
        Coord<DIM> relativeCoord = absoluteCoord - origin;
        if (TOPOLOGICALLY_CORRECT) 
            relativeCoord = 
                Topology::normalize(relativeCoord, topoDimensions);
        return delegate[relativeCoord];
    }

    inline const CELL_TYPE& operator[](const Coord<DIM>& absoluteCoord) const
    {
        return (const_cast<DisplacedGrid&>(*this))[absoluteCoord];
    }

    virtual CELL_TYPE& at(const Coord<DIM>& coord)
    {
        return (*this)[coord];
    }

    virtual const CELL_TYPE& at(const Coord<DIM>& coord) const
    {
        return (*this)[coord];
    }

    virtual CELL_TYPE& atEdge()
    {
        return getEdgeCell();
    }

    virtual const CELL_TYPE& atEdge() const
    {
        return getEdgeCell();
    }

    template<typename  GRID_TYPE>
    inline void paste(const GRID_TYPE& grid, const Region<DIM>& region)
    {
        for (StreakIterator<DIM> i = region.beginStreak(); i != region.endStreak(); ++i) {
            const CELL_TYPE *start = &grid[i->origin];
            std::copy(start, start + i->length(), &(*this)[i->origin]);
        }
    }

    inline void pasteGridBase(const GridBase<CELL_TYPE, DIM>& grid, const Region<DIM>& region)
    {
        for (StreakIterator<DIM> i = region.beginStreak(); i != region.endStreak(); ++i) {
            const CELL_TYPE *start = &grid.at(i->origin);
            std::copy(start, start + i->length(), &(*this)[i->origin]);
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

    inline MyCoordMap getNeighborhood(const Coord<DIM>& center)
    {
        Coord<DIM> relativeCoord = center - origin;
        if (TOPOLOGICALLY_CORRECT) 
            relativeCoord = 
                Topology::normalize(relativeCoord, topoDimensions);
        return MyCoordMap(relativeCoord, &delegate);
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
        message << "DisplacedGrid\n"
                << "  origin: " << origin << "\n"
                << "  delegate:\n"
                << delegate;
        return message.str();
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
