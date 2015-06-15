#ifndef LIBGEODECOMP_STORAGE_PROXYGRID_H
#define LIBGEODECOMP_STORAGE_PROXYGRID_H

#include <libgeodecomp/storage/gridbase.h>

namespace LibGeoDecomp {

/**
 * This class provides a (reduced) view of another grid. This is
 * helpful if, e.g., a Simulator is internally padding a grid but this
 * implementation details shouldn't be given up to every initializer.
 */
template<typename CELL, int DIM>
class ProxyGrid : public GridBase<CELL, DIM>
{
public:
    ProxyGrid(GridBase<CELL, DIM>& delegate, const CoordBox<DIM>& viewBox) :
        delegate(delegate),
        viewBox(viewBox)
    {}

    virtual void set(const Coord<DIM>& coord, const CELL& cell)
    {
        delegate.set(coord, cell);
    }

    virtual void set(const Streak<DIM>& streak, const CELL *cells)
    {
        delegate.set(streak, cells);
    }

    virtual CELL get(const Coord<DIM>& coord) const
    {
        return delegate.get(coord);
    }

    virtual void get(const Streak<DIM>& streak, CELL *cells) const
    {
        delegate.get(streak, cells);
    }

    virtual void setEdge(const CELL& cell)
    {
        delegate.setEdge(cell);
    }

    virtual const CELL& getEdge() const
    {
        return delegate.getEdge();
    }

    virtual CoordBox<DIM> boundingBox() const
    {
        return viewBox;
    }

    // fixme: needs test
    void saveMemberImplementation(char *s, const Selector<CELL>& selector, const Region<DIM>& region) const
    {
        delegate.saveMemberImplementation(s, selector, region);
    }

    // fixme: needs test
    void loadMemberImplementation(const char *s, const Selector<CELL>& selector, const Region<DIM>& region)
    {
        delegate.loadMemberImplementation(s, selector, region);
    }

private:
    GridBase<CELL, DIM>& delegate;
    CoordBox<DIM> viewBox;
};

}

#endif
