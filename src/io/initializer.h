#ifndef LIBGEODECOMP_IO_INITIALIZER_H
#define LIBGEODECOMP_IO_INITIALIZER_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/gridbase.h>

#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
#include <boost/serialization/base_object.hpp>
#endif


namespace LibGeoDecomp {

/**
 * The initializer sets up the initial state of the grid. For this a
 * Simulator will invoke Initializer::grid(). Keep in mind that grid()
 * might be called multiple times and that for parallel runs each
 * Initializer will be responsible just for a sub-cuboid of the whole
 * grid.
 */
template<typename CELL>
class Initializer
{
public:
    typedef typename APITraits::SelectTopology<CELL>::Value Topology;
    static const unsigned NANO_STEPS = APITraits::SelectNanoSteps<CELL>::VALUE;

#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
    friend class boost::serialization::access;
#endif
    static const int DIM = Topology::DIM;

    /**
     * initializes all cells of the grid at target
     */
    virtual void grid(GridBase<CELL, Topology::DIM> *target) = 0;

    virtual ~Initializer()
    {}

    virtual CoordBox<DIM> gridBox()
    {
        return CoordBox<DIM>(Coord<DIM>(), gridDimensions());
    }

    virtual Coord<DIM> gridDimensions() const = 0;
    virtual unsigned maxSteps() const = 0;
    virtual unsigned startStep() const = 0;

private:

#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {}
#endif
};

}

#endif
