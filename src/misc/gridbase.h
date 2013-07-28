#ifndef LIBGEODECOMP_MISC_GRIDBASE_H
#define LIBGEODECOMP_MISC_GRIDBASE_H

#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/coordbox.h>

namespace LibGeoDecomp {

/**
 * This is an abstract base class for all grid classes. It's generic
 * because all methods are virtual, but not very efficient -- for the
 * same reason.
 */
template<typename CELL, int DIMENSIONS>
class GridBase
{
public:
    typedef CELL CellType;
    const static int DIM = DIMENSIONS;

    virtual ~GridBase()
    {}

    // fixme: reove these
    virtual CELL& at(const Coord<DIM>&) = 0;
    virtual const CELL& at(const Coord<DIM>&) const = 0;
    virtual CELL& atEdge() = 0;
    virtual const CELL& atEdge() const = 0;
    // fixme: add functions for getting/setting streak of cells. use these in mpiio.h, writers, steerers, mpilayer...
    virtual void set(const Coord<DIM>& coord, const CELL& cell)
    {
        at(coord) = cell;
    }

    virtual CELL get(const Coord<DIM>& coord) const
    {
        return at(coord);
    }

    virtual void setEdge(const CELL& cell)
    {
        atEdge() = cell;
    }

    virtual const CELL& getEdge() const
    {
        return atEdge();
    }

    virtual CoordBox<DIM> boundingBox() const = 0;
};

}

#endif
