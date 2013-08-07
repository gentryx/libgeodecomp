#ifndef LIBGEODECOMP_MISC_CELLAPITRAITS_H
#define LIBGEODECOMP_MISC_CELLAPITRAITS_H

#include <libgeodecomp/misc/displacedgrid.h>
#include <libgeodecomp/misc/soagrid.h>

namespace LibGeoDecomp {

namespace API {

// deduce a CELL's optimum grid type
template<typename CELL, typename HAS_SOA = void>
class SelectGridType
{
public:
    typedef DisplacedGrid<CELL, typename CELL::Topology> Type;
};

template<typename CELL>
class SelectGridType<CELL, typename CELL::API::HasSoA>
{
public:
    typedef SoAGrid<CELL, typename CELL::Topology> Type;
};

/**
 * Use this qualifier in a cell's API to hint that it supports a
 * Struct of Arrays memory layout.
 */
class SupportsSoA
{
public:
    typedef void HasSoA;
};

}

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

