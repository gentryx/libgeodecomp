#ifndef _libgeodecomp_misc_cellapitraits_h_
#define _libgeodecomp_misc_cellapitraits_h_

namespace LibGeoDecomp {

/**
 * is used to specify which neighborhood types are supported by a
 * given cell or Simulator/Stepper. This is necessary as the different
 * neighborhood implementations vary greatly in performance (depending
 * on the hardware) and some may even be incompatible with certain
 * models (e.g. when the relative coordinates for neighbor accesses
 * are not known at compile time).
 */
class CellAPITraits
{
public:
    /**
     * If a cell's API specifier derives only from this class and no
     * other class, it means that the class is using the classic
     * (default) API.
     */
    class Base
    {};

    /**
     * Fixed means that a given model may only use FixedCoord to
     * address neighbors, which allows us to do significant compile
     * time optimizations.
     */
    class Fixed : public Base
    {
    };

    /**
     * indicates that the model may not only update a single cell, but
     * a linear sequence cells within the grid. 
     */
    class Line : public Base
    {
    };
};

}

#endif

