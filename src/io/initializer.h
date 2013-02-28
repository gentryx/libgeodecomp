#ifndef _libgeodecomp_io_initializer_h_
#define _libgeodecomp_io_initializer_h_

#include <libgeodecomp/misc/gridbase.h>

#include <boost/serialization/base_object.hpp>

namespace LibGeoDecomp {

template<typename CELL>
class Initializer
{
public:
    static const int DIMENSIONS = CELL::Topology::DIMENSIONS;

    /**
     * initializes all cells of the grid at @a target 
     */
    virtual void grid(GridBase<CELL, CELL::Topology::DIMENSIONS> *target) =0;

    virtual ~Initializer() 
    {}

    virtual CoordBox<DIMENSIONS> gridBox()
    {
        return CoordBox<DIMENSIONS>(Coord<DIMENSIONS>(), gridDimensions());
    }

    virtual Coord<DIMENSIONS> gridDimensions() const = 0;
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
