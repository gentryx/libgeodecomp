#ifndef LIBGEODECOMP_STORAGE_PROXYGRID_H
#define LIBGEODECOMP_STORAGE_PROXYGRID_H

#include <libgeodecomp/storage/gridbase.h>

namespace LibGeoDecomp {

/**
 * This class provides a (reduced) view of another grid. This is
 * helpful if, e.g., a Simulator is internally padding a grid but this
 * implementation details shouldn't be given up to every initializer.
 */
template<typename CELL, int DIM, typename WEIGHT_TYPE = double>
class ProxyGrid : public GridBase<CELL, DIM, WEIGHT_TYPE>
{
public:
    ProxyGrid(GridBase<CELL, DIM, WEIGHT_TYPE> *delegate, const CoordBox<DIM>& viewBox) :
        delegate(delegate),
        viewBox(viewBox)
    {}

    virtual void resize(const CoordBox<DIM>& newBox)
    {
        viewBox = newBox;
    }

    virtual void set(const Coord<DIM>& coord, const CELL& cell)
    {
        delegate->set(coord, cell);
    }

    virtual void set(const Streak<DIM>& streak, const CELL *cells)
    {
        delegate->set(streak, cells);
    }

    virtual CELL get(const Coord<DIM>& coord) const
    {
        return delegate->get(coord);
    }

    virtual void get(const Streak<DIM>& streak, CELL *cells) const
    {
        delegate->get(streak, cells);
    }

    virtual void setEdge(const CELL& cell)
    {
        delegate->setEdge(cell);
    }

    virtual const CELL& getEdge() const
    {
        return delegate->getEdge();
    }

    virtual CoordBox<DIM> boundingBox() const
    {
        return viewBox;
    }

    using GridBase<CELL, DIM, WEIGHT_TYPE>::saveRegion;
    using GridBase<CELL, DIM, WEIGHT_TYPE>::loadRegion;

    void saveRegion(
        std::vector<CELL> *buffer,
        const Region<DIM>& region,
        const Coord<DIM>& offset = Coord<DIM>()) const
    {
        delegate->saveRegion(buffer, region, offset);
    }

#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
    void saveRegion(
        std::vector<char> *buffer,
        const Region<DIM>& region,
        const Coord<DIM>& offset = Coord<DIM>()) const
    {
        delegate->saveRegion(buffer, region, offset);
    }
#endif

    void loadRegion(
        const std::vector<CELL>& buffer,
        const Region<DIM>& region,
        const Coord<DIM>& offset = Coord<DIM>())
    {
        delegate->loadRegion(buffer, region, offset);
    }

#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
    void loadRegion(
        const std::vector<char>& buffer,
        const Region<DIM>& region,
        const Coord<DIM>& offset = Coord<DIM>())
    {
        delegate->loadRegion(buffer, region, offset);
    }
#endif

    void saveMemberImplementation(
        char *target,
        MemoryLocation::Location targetLocation,
        const Selector<CELL>& selector,
        const typename Region<DIM>::StreakIterator& begin,
        const typename Region<DIM>::StreakIterator& end) const
    {
        delegate->saveMemberImplementation(target, targetLocation, selector, begin, end);
    }

    void loadMemberImplementation(
        const char *source,
        MemoryLocation::Location sourceLocation,
        const Selector<CELL>& selector,
        const typename Region<DIM>::StreakIterator& begin,
        const typename Region<DIM>::StreakIterator& end)
    {
        delegate->loadMemberImplementation(source, sourceLocation, selector, begin, end);
    }

private:
    GridBase<CELL, DIM> *delegate;
    CoordBox<DIM> viewBox;
};

}

#endif
