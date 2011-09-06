#ifndef _libgeodecomp_misc_gridbase_h_
#define _libgeodecomp_misc_gridbase_h_

#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/coordbox.h>

namespace LibGeoDecomp {

/**
 * This is an abstract base class for all grid classes. It's generic
 * because all methods are virtual, but not very efficient -- for the
 * same reason.
 */
template<typename CELL, int DIM>
class GridBase
{
public:
    virtual CELL& at(const Coord<DIM>&) =0;
    virtual const CELL& at(const Coord<DIM>&) const =0;
    virtual CELL& atEdge() =0;
    virtual const CELL& atEdge() const =0;
    virtual CoordBox<DIM> boundingBox() const =0;
};

}

#endif
