#ifndef LIBGEODECOMP_IO_SIMPLEINITIALIZER_H
#define LIBGEODECOMP_IO_SIMPLEINITIALIZER_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/io/initializer.h>

namespace LibGeoDecomp {

/**
 * This convenience class implements some straightforward functions of
 * Initializer. Generally users will only need to implement grid().
 */
template<typename CELL_TYPE>
class SimpleInitializer : public Initializer<CELL_TYPE>
{
public:
    friend class Serialization;
    typedef typename Initializer<CELL_TYPE>::Topology Topology;
    const static int DIM = Topology::DIM;

    // fixme: writers AND initializers should have a clone() function, preferably implemented via CRTP
    SimpleInitializer(
        const Coord<DIM>& dimensions,
        const unsigned steps = 300) :
        dimensions(dimensions),
        steps(steps)
    {}

    Coord<DIM> gridDimensions() const
    {
        return dimensions;
    }

    unsigned maxSteps() const
    {
        return steps;
    }

    unsigned startStep() const
    {
        return 0;
    }

protected:
    Coord<DIM> dimensions;
    unsigned steps;
};

}

#endif
