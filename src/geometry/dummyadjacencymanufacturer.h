#ifndef LIBGEODECOMP_GEOMETRY_DUMMYADJACENCYMANUFACTURER_H
#define LIBGEODECOMP_GEOMETRY_DUMMYADJACENCYMANUFACTURER_H

#include <libgeodecomp/geometry/adjacencymanufacturer.h>
#include <libgeodecomp/geometry/regionbasedadjacency.h>

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
    virtual boost::shared_ptr<Adjacency> getAdjacency(const Region<DIM>& region) const
    {
        return boost::make_shared<RegionBasedAdjacency>(RegionBasedAdjacency());
    }
};

}

#endif
