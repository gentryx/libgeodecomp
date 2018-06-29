#ifndef LIBGEODECOMP_STORAGE_DISPLACEDGRID_H
#define LIBGEODECOMP_STORAGE_DISPLACEDGRID_H

#include <libgeodecomp/geometry/coordbox.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/storage/grid.h>

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
#include <libgeodecomp/misc/cudaboostworkaround.h>
#include <libgeodecomp/communication/boostserialization.h>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#endif

#ifdef LIBGEODECOMP_WITH_HPX
#include <libgeodecomp/misc/cudaboostworkaround.h>
#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>
#endif

namespace LibGeoDecomp {

#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4820 )
#endif

namespace DisplacedGridHelpers {

template<class TOPOLOGY, bool TOPOLOGICALLY_CORRECT>
class NormalizingIterator : public RegionStreakIterator<TOPOLOGY::DIM, Region<TOPOLOGY::DIM> >
{
public:
    static const int DIM = TOPOLOGY::DIM;
    inline NormalizingIterator(
        const RegionStreakIterator<DIM, Region<DIM> >& iter,
        const Coord<DIM>& origin,
        const Coord<DIM>& topoDimensions) :
        RegionStreakIterator<DIM, Region<DIM> >(iter),
        origin(origin),
        topoDimensions(topoDimensions)
    {
        normalize();
    }

    inline void operator++()
    {
        RegionStreakIterator<DIM, Region<DIM> >::operator++();
        normalize();
    }

    inline const Streak<DIM>& operator*() const
    {
        return normalizedStreak;
    }

    inline const Streak<DIM> *operator->() const
    {
        return &normalizedStreak;
    }

private:
    Streak<DIM> normalizedStreak;
    const Coord<DIM>& origin;
    const Coord<DIM>& topoDimensions;

    inline void normalize()
    {
        normalizedStreak = this->streak;
        normalizedStreak.origin -= origin;

        if (TOPOLOGICALLY_CORRECT) {
            normalizedStreak.origin = TOPOLOGY::normalize(normalizedStreak.origin, topoDimensions);
        }

        normalizedStreak.endX = normalizedStreak.origin.x() + this->streak.length();
    }
};

}

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

    void saveRegion(
        std::vector<CELL_TYPE> *buffer,
        const Region<DIM>& region,
        const Coord<DIM>& offset = Coord<DIM>()) const
    {
        typedef DisplacedGridHelpers::NormalizingIterator<TOPOLOGY, TOPOLOGICALLY_CORRECT> NormalizingIterator;
        NormalizingIterator iter(region.beginStreak(offset), origin, topoDimensions);
        delegate.saveRegionImplementation(buffer, iter, region.endStreak(offset - origin));
    }

#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
    void saveRegion(
        std::vector<char> *buffer,
        const Region<DIM>& region,
        const Coord<DIM>& offset = Coord<DIM>()) const
    {
        typedef typename APITraits::SelectBoostSerialization<CELL_TYPE>::Value Trait;
        delegate.saveRegionImplementation(buffer, region.beginStreak(offset - origin), region.endStreak(offset - origin), Trait());
    }
#endif

    void loadRegion(
        const std::vector<CELL_TYPE>& buffer,
        const Region<DIM>& region,
        const Coord<DIM>& offset = Coord<DIM>())
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

#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
    void loadRegion(
        const std::vector<char>& buffer,
        const Region<DIM>& region,
        const Coord<DIM>& offset = Coord<DIM>())
    {
        typedef typename APITraits::SelectBoostSerialization<CELL_TYPE>::Value Trait;
        delegate.loadRegionImplementation(buffer, region.beginStreak(offset - origin), region.endStreak(offset - origin), Trait());
    }
#endif

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
        const typename Region<DIM>::StreakIterator& begin,
        const typename Region<DIM>::StreakIterator& end) const
    {
        typedef DisplacedGridHelpers::NormalizingIterator<TOPOLOGY, TOPOLOGICALLY_CORRECT> NormalizingIterator;

        delegate.saveMemberImplementationTemplated(
            target,
            targetLocation,
            selector,
            NormalizingIterator(begin, origin, topoDimensions),
            end);
    }

    void loadMemberImplementation(
        const char *source,
        MemoryLocation::Location sourceLocation,
        const Selector<CELL_TYPE>& selector,
        const typename Region<DIM>::StreakIterator& begin,
        const typename Region<DIM>::StreakIterator& end)
    {
        typedef DisplacedGridHelpers::NormalizingIterator<TOPOLOGY, TOPOLOGICALLY_CORRECT> NormalizingIterator;

        delegate.loadMemberImplementationTemplated(
            source,
            sourceLocation,
            selector,
            NormalizingIterator(begin, origin, topoDimensions),
            end);
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
