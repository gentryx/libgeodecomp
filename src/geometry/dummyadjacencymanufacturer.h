#ifndef LIBGEODECOMP_GEOMETRY_DUMMYADJACENCYMANUFACTURER_H
#define LIBGEODECOMP_GEOMETRY_DUMMYADJACENCYMANUFACTURER_H

#include <libgeodecomp/geometry/adjacencymanufacturer.h>
#include <libgeodecomp/geometry/regionbasedadjacency.h>
#include <libgeodecomp/misc/sharedptr.h>

namespace LibGeoDecomp {

/**
 * This dummy class will always return empty Adjacencies. Useful for
 * setting up domain decompostions where no adjacency is required
 * (e.g. regular grids) or when the initial grid is empty.
 */
template<int DIM>
class DummyAdjacencyManufacturer : public AdjacencyManufacturer<DIM>
{
public:
    typedef typename AdjacencyManufacturer<DIM>::AdjacencyPtr AdjacencyPtr;

    virtual AdjacencyPtr getAdjacency(const Region<DIM>& region) const
    {
        return AdjacencyPtr(new RegionBasedAdjacency());
    }
};

}

#endif
