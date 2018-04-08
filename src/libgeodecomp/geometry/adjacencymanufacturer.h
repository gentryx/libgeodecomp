#ifndef LIBGEODECOMP_GEOMETRY_ADJACENCYMANUFACTURER_H
#define LIBGEODECOMP_GEOMETRY_ADJACENCYMANUFACTURER_H

// Kill warning 4514 in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <stdexcept>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

#include <libgeodecomp/geometry/adjacency.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/geometry/regionbasedadjacency.h>
#include <libgeodecomp/misc/sharedptr.h>

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
    typedef typename SharedPtr<Adjacency>::Type AdjacencyPtr;

    virtual ~AdjacencyManufacturer()
    {}

    /**
     * yields an Adjacency object with all outgoing edges of the node
     * set in the given Region.
     */
    virtual AdjacencyPtr getAdjacency(const Region<DIM>& region) const = 0;

    /**
     * Same a getAdjacency, but returns all edges pointing to the
     * nodes in the given Region. Also inverts the direction of the
     * edges. Used in the PartitionManager to grow inverse rims around
     * a region. See UnstructuredTestInitializer for an example.
     */
    virtual AdjacencyPtr getReverseAdjacency(const Region<DIM>& region) const = 0;
};

}

#endif
