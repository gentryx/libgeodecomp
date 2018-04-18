#ifndef LIBGEODECOMP_STORAGE_DISPLACEDGRID_H
#define LIBGEODECOMP_STORAGE_DISPLACEDGRID_H

#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/storage/grid.h>

namespace LibGeoDecomp {

#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4710 4711 4820 )
#endif

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
    typedef Grid<CELL_TYPE, TOPOLOGY> Delegate;
    typedef CoordMap<CELL_TYPE, Delegate> CoordMapType;

    using GridBase<CELL_TYPE, TOPOLOGY::DIM>::loadRegion;
    using GridBase<CELL_TYPE, TOPOLOGY::DIM>::saveRegion;
    using GridBase<CELL_TYPE, TOPOLOGY::DIM>::topoDimensions;

    explicit DisplacedGrid(
        const CoordBox<DIM>& box = CoordBox<DIM>(),
        const CELL_TYPE& defaultCell = CELL_TYPE(),
        const CELL_TYPE& edgeCell = CELL_TYPE(),
        const Coord<DIM>& topologicalDimensions = Coord<DIM>()) :
        GridBase<CELL_TYPE, TOPOLOGY::DIM>(topologicalDimensions),
        delegate(box.dimensions, defaultCell, edgeCell),
        origin(box.origin)
    {}

    explicit DisplacedGrid(
        const Region<DIM>& region,
        const CELL_TYPE& defaultCell = CELL_TYPE(),
        const CELL_TYPE& edgeCell = CELL_TYPE(),
        const Coord<DIM>& topologicalDimensions = Coord<DIM>()) :
        GridBase<CELL_TYPE, TOPOLOGY::DIM>(topologicalDimensions),
        delegate(region.boundingBox().dimensions, defaultCell, edgeCell),
        origin(region.boundingBox().origin)
    {}

    explicit DisplacedGrid(
        const Delegate& grid,
        const Coord<DIM>& origin=Coord<DIM>()) :
        delegate(grid),
        origin(origin)
    {}

    /**
     * Return a pointer to the underlying data storage. Use with care!
     */
    inline
    CELL_TYPE *data()
    {
        return delegate.data();
    }

    /**
     * Return a const pointer to the underlying data storage. Use with
     * care!
     */
    inline
    const CELL_TYPE *data() const
    {
        return delegate.data();
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

    inline const Coord<DIM>& getDimensions() const
    {
        return delegate.getDimensions();
    }

    virtual CoordBox<DIM> boundingBox() const
    {
        return CoordBox<DIM>(origin, delegate.getDimensions());
    }

    void saveRegion(std::vector<CELL_TYPE> *buffer, const Region<DIM>& region, const Coord<DIM>& offset = Coord<DIM>()) const
    {
        CELL_TYPE *source = buffer->data();
        typename Region<DIM>::StreakIterator end = region.endStreak(offset);
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(offset); i != end; ++i) {
            Coord<DIM> relativeCoord = i->origin - origin;
            if (TOPOLOGICALLY_CORRECT) {
                relativeCoord = Topology::normalize(relativeCoord, topoDimensions);
            }

            Streak<DIM> streak(relativeCoord, relativeCoord.x() + i->length());

            delegate.get(streak, source);
            source += i->length();
        }
    }

    void loadRegion(const std::vector<CELL_TYPE>& buffer, const Region<DIM>& region, const Coord<DIM>& offset = Coord<DIM>())
    {
        const CELL_TYPE *source = buffer.data();
        typename Region<DIM>::StreakIterator end = region.endStreak(offset);
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(offset); i != end; ++i) {
            Coord<DIM> relativeCoord = i->origin - origin;
            if (TOPOLOGICALLY_CORRECT) {
                relativeCoord = Topology::normalize(relativeCoord, topoDimensions);
            }

            Streak<DIM> streak(relativeCoord, relativeCoord.x() + i->length());

            delegate.set(streak, source);
            source += i->length();
        }
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
        char *target,
        MemoryLocation::Location targetLocation,
        const Selector<CELL_TYPE>& selector,
        const Region<DIM>& region) const
    {
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            selector.copyMemberOut(
                &(*this)[i->origin],
                MemoryLocation::HOST,
                target,
                targetLocation,
                std::size_t(i->length()));
            target += selector.sizeOfExternal() * i->length();
        }
    }

    void loadMemberImplementation(
        const char *source,
        MemoryLocation::Location sourceLocation,
        const Selector<CELL_TYPE>& selector,
        const Region<DIM>& region)
    {
        for (typename Region<DIM>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
            selector.copyMemberIn(
                source,
                sourceLocation,
                &(*this)[i->origin],
                MemoryLocation::HOST,
                std::size_t(i->length()));
            source += selector.sizeOfExternal() * i->length();
        }
    }


private:
    Delegate delegate;
    Coord<DIM> origin;
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

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
