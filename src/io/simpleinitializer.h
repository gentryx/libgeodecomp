#ifndef LIBGEODECOMP_IO_SIMPLEINITIALIZER_H
#define LIBGEODECOMP_IO_SIMPLEINITIALIZER_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/io/initializer.h>

#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
#include <boost/serialization/base_object.hpp>
#endif

namespace LibGeoDecomp {

template<typename CELL_TYPE>
class SimpleInitializer : public Initializer<CELL_TYPE>
{
public:
    const static int DIM = CELL_TYPE::Topology::DIM;

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

#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {
        ar & boost::serialization::base_object<Initializer<CELL_TYPE> >(*this);
        ar & dimensions;
        ar & steps;
    }
#endif

protected:
    Coord<DIM> dimensions;
    unsigned steps;
};

}

#endif
