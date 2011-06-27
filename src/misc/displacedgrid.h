#ifndef _libgeodecomp_misc_displacedgrid_h_
#define _libgeodecomp_misc_displacedgrid_h_

#include <libgeodecomp/misc/coordbox.h>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/misc/region.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE, typename TOPOLOGY=Topologies::Cube<2>::Topology>
class DisplacedGrid : public GridBase<CELL_TYPE, TOPOLOGY::DIMENSIONS>
{
public:
    const static int DIMENSIONS = TOPOLOGY::DIMENSIONS;

    typedef CELL_TYPE CellType;
    typedef TOPOLOGY Topology;
    typedef typename boost::detail::multi_array::sub_array<CELL_TYPE, 1> RowRef;
    typedef typename boost::detail::multi_array::const_sub_array<CELL_TYPE, 1> ConstRowRef;
    typedef typename boost::multi_array<CELL_TYPE, DIMENSIONS>::index Index;
    typedef Grid<CELL_TYPE, TOPOLOGY> Delegate;
    typedef CoordMap<CELL_TYPE, Delegate> MyCoordMap;

    DisplacedGrid(
        const CoordBox<DIMENSIONS>& box = CoordBox<DIMENSIONS>(),
        const CELL_TYPE& _defaultCell=CELL_TYPE()) :
        delegate(box.dimensions, _defaultCell),
        origin(box.origin)
    {}

    DisplacedGrid(const Delegate& _grid,
                  const Coord<DIMENSIONS>& _origin=Coord<DIMENSIONS>()) :
        delegate(_grid),
        origin(_origin)
    {}

    inline CELL_TYPE *baseAddress()
    {
        return delegate.baseAddress();
    }

    inline const CELL_TYPE *baseAddress() const
    {
        return delegate.baseAddress();
    }

    inline const Coord<DIMENSIONS>& getOrigin() const
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

    inline void resize(const CoordBox<DIMENSIONS>& box)
    {
        delegate.resize(box.dimensions);
        origin = box.origin;
    }

    inline CELL_TYPE& operator[](const Coord<DIMENSIONS>& absoluteCoord)
    {
        Coord<DIMENSIONS> relativeCoord = absoluteCoord - origin;
        return delegate[relativeCoord];
    }

    inline const CELL_TYPE& operator[](const Coord<DIMENSIONS>& absoluteCoord) const
    {
        Coord<DIMENSIONS> relativeCoord = absoluteCoord - origin;
        return delegate[relativeCoord];
    }

    inline const ConstRowRef operator[](const Index& y) const
    {
        return delegate[y - origin.y];
    }

    inline RowRef operator[](const Index& y) 
    {
        return delegate[y - origin.y];
    }

    virtual CELL_TYPE& at(const Coord<DIMENSIONS>& coord)
    {
        return (*this)[coord];
    }

    virtual const CELL_TYPE& at(const Coord<DIMENSIONS>& coord) const
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
    inline void paste(const GRID_TYPE& grid, const Region<DIMENSIONS>& region)
    {
        for (StreakIterator<DIMENSIONS> i = region.beginStreak(); i != region.endStreak(); ++i) {
            const CELL_TYPE *start = &grid[i->origin];
            std::copy(start, start + i->length(), &(*this)[i->origin]);
        }
    }

    inline const Coord<DIMENSIONS>& getDimensions() const
    {
        return delegate.getDimensions();
    }

    virtual CoordBox<DIMENSIONS> boundingBox() const
    {
        return CoordBox<DIMENSIONS>(origin, delegate.getDimensions());
    }

    inline MyCoordMap getNeighborhood(const Coord<DIMENSIONS>& center)
    {
        return MyCoordMap(center - origin, &delegate);
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
    Coord<DIMENSIONS> origin;
};

};

template<typename _CharT, typename _Traits, typename _CellT, typename _Topology>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const LibGeoDecomp::DisplacedGrid<_CellT, _Topology>& grid)
{
    __os << grid.toString();
    return __os;
}

#endif
