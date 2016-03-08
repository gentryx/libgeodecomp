#ifndef LIBGEODECOMP_GEOMETRY_ADJACENCYMANUFACTURER_H
#define LIBGEODECOMP_GEOMETRY_ADJACENCYMANUFACTURER_H

#include <stdexcept>
#include <libgeodecomp/geometry/adjacency.h>
#include <libgeodecomp/geometry/region.h>

namespace LibGeoDecomp {

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

    virtual boost::shared_ptr<Adjacency> getAdjacency(const Region<DIM>& region) const = 0;
};

}

#endif
