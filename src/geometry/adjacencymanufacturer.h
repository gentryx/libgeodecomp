#ifndef LIBGEODECOMP_GEOMETRY_ADJACENCYMANUFACTURER_H
#define LIBGEODECOMP_GEOMETRY_ADJACENCYMANUFACTURER_H

#include <stdexcept>
#include <libgeodecomp/geometry/adjacency.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/geometry/regionbasedadjacency.h>

namespace LibGeoDecomp {

class RegionBasedAdjacency;

/**
 * Wrapper to separate Adjacency creation from the Cell template
 * parameter type in Initializers.
 */
template<int DIM>
class AdjacencyManufacturer
{
public:
    virtual ~AdjacencyManufacturer()
    {}

    /**
     * yields an Adjacency object with all outgoing edges of the node
     * set in the given Region.
     */
    virtual boost::shared_ptr<Adjacency> getAdjacency(const Region<DIM>& region) const = 0;

    /**
     * Same a getAdjacency, but returns all edges pointing to the
     * nodes in the given Region. Also inverts the direction of the
     * edges. Used in the PartitionManager to grow inverse rims around
     * a region. See UnstructuredTestInitializer for an example.
     */
    virtual boost::shared_ptr<Adjacency> getReverseAdjacency(const Region<DIM>& region) const = 0;
};

}

#endif
