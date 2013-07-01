#ifndef LIBGEODECOMP_IO_INITIALIZER_H
#define LIBGEODECOMP_IO_INITIALIZER_H

#include <libgeodecomp/misc/gridbase.h>

#include <boost/serialization/base_object.hpp>

namespace LibGeoDecomp {

template<typename CELL>
class Initializer
{
public:
    static const int DIM = CELL::Topology::DIM;

    /**
     * initializes all cells of the grid at @a target
     */
    virtual void grid(GridBase<CELL, CELL::Topology::DIM> *target) =0;

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
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive & ar, unsigned)
    {}
};

}

#endif
