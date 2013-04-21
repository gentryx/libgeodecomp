#ifndef LIBGEODECOMP_IO_SIMPLEINITIALIZER_H
#define LIBGEODECOMP_IO_SIMPLEINITIALIZER_H

#include <libgeodecomp/io/initializer.h>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class SimpleInitializer : public Initializer<CELL_TYPE>
{
public:
    const static int DIM = CELL_TYPE::Topology::DIM;

    SimpleInitializer(
        const Coord<DIM>& _dimensions,
        const unsigned& _steps = 300) :
        dimensions(_dimensions), steps(_steps)
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
