#ifndef _libgeodecomp_io_simpleinitializer_h_
#define _libgeodecomp_io_simpleinitializer_h_

#include <libgeodecomp/io/initializer.h>

#include <boost/serialization/base_object.hpp>

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class SimpleInitializer : public Initializer<CELL_TYPE>
{
public:
    const static int DIMENSIONS = CELL_TYPE::Topology::DIMENSIONS;

    SimpleInitializer(
        const Coord<DIMENSIONS>& _dimensions,
        const unsigned& _steps = 300) :
        dimensions(_dimensions), steps(_steps)
    {}

    Coord<DIMENSIONS> gridDimensions() const
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
    
    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
        ar & boost::serialization::base_object<Initializer<CELL_TYPE> >(*this);
    }

protected:
    Coord<DIMENSIONS> dimensions;
    unsigned steps;
};

};

#endif
