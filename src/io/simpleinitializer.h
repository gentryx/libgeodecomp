#ifndef LIBGEODECOMP_IO_SIMPLEINITIALIZER_H
#define LIBGEODECOMP_IO_SIMPLEINITIALIZER_H

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
    typedef typename Initializer<CELL_TYPE>::Topology Topology;
    const static int DIM = Topology::DIM;

    explicit SimpleInitializer(
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
