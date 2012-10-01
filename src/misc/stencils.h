#ifndef _libgeodecomp_misc_stencils_h_
#define _libgeodecomp_misc_stencils_h_

#include <libgeodecomp/misc/coord.h>

namespace LibGeoDecomp {

/**
 * is a collection of stencil shapes which cells may use to describe
 * the neighborhood they need during their updates. It also contains
 * some utility classes to ease writing stencil-shape-agnostic code
 * (i.e. bundle stencil-specific code in Stencils).
 *
 * We'll use a number of shortcuts to designate the different spatial
 * directions. The first table names the cardinal directions. They may
 * be combined in MSB-first order to form non-cardinal directions. For
 * instance B would be BOTTOM, L is LEFT, and BL would correspond to
 * BOTTOM LEFT, or the relative coordinate (-1, 1, 0).
 *
 * Flag   Short Axis
 * ------ ----- ----
 * WEST    W    X-
 * EAST    E    X+
 * TOP     T    Y-
 * BOTTOM  B    Y+
 * SOUTH   S    Z-
 * NORTH   N    Z+
 * CENTER  C    0

 */
class Stencils
{
private:
    friend class StencilsTest;

    template<int BASE, int EXP>
    class Power;

    template<template<int, int> class ADDEND, int INDEX, int DIM>
    class Sum;

    /**
     * Required for calculating the volume of a VonNeumann's stencil correctly.
     */
    template<int RADIUS, int DIM>
    class VonNeumannHelper;

public:

    /**
     * The classic Moore neighborhood contains all cells whose spatial
     * distance to the orign (i.e. the current cell) -- as measured by
     * the maximum norm -- is smaller or equal to RADIUS.
     */
    template<int DIM, int RADIUS>
    class Moore
    {
    public:
        static const int VOLUME = Power<RADIUS * 2 + 1, DIM>::VALUE;

        template<int INDEX>
        class Coords;
    };

    /**
     * The VonNeumann neighborhood is probably as well known as the
     * Moore neighborhood, but most commonly only used with a RADIUS
     * equal to 1. It replaces the maximum norm with the Manhattan
     * distance.
     */
    template<int DIM, int RADIUS>
    class VonNeumann
    {
    public:
        static const int VOLUME = 
            VonNeumann<DIM - 1, RADIUS>::VOLUME +
            2 * Sum<VonNeumannHelper, RADIUS - 1, DIM - 1>::VALUE;

        template<int INDEX>
        class Coords;
    };

    template<int RADIUS>
    class VonNeumann<0, RADIUS>
    {
    public:
        static const int VOLUME = 1;
    };

    /**
     * This neighborhood was dubbed "Cross" by Prof. Dietmar Fey
     * because of its shape. It contains all coordinates whose maximum
     * norms equal their Manhattan distances and which are within the
     * stencil radius accoring to either norm. (Yes, that's a
     * complicated, but formally correct way to describe an
     * n-dimensional cross.)
     */
    template<int DIM, int RADIUS>
    class Cross
    {
    public:
        static const int VOLUME = 1 + 2 * RADIUS * DIM;

        template<int INDEX>
        class Coords;
    };

    /**
     * This is a utility class to aid in adressing all neighboring
     * cells which are packed in a linear array. Examples:
     *
     * 1D Moore, von Neumann:
     *
     * {W, C, E}
     *
     * 2D Moore:
     *
     * {TW, T, TE, W, C, E, BW, B, BE}
     *
     * 3D Moore:
     *
     * {STW, ST, STE, SW, S, SE, SBW, SB, SBE,
     *  TW, T, TE, W, C, E, BW, B, BE,
     *  NTW, NT, NTE, NW, N, NE, NBW, NB, NBE}
     *
     * 2D von Neumann:
     *
     * {T, W, C, E, B}
     *
     * 3D von Neumann:
     * 
     * {S, T, W, C, E, B, N}
     *
     * From these we can deduce the canonical offsets within the array
     * for relative coordinates:
     *
     * - Moore offsets (any dimension)
     *   - X offset: X * 1, fix: 1
     *   - Y offset: Y * 3, fix: 3
     *   - Z offset: Z * 9, fix: 9
     *   - dimension-specific offset:
     *     D == 1 -> 1
     *     D == 2 -> 4
     *     D == 3 -> 13
     *
     * - von Neumann offsets (any dimension)
     *   - X offset: X * 1, 1
     *   - Y offset: Y * 2, 1
     *   - Z offset: Z * 3, 1
     *   - dimension-specific offset:
     *     D == 1 -> 1
     *     D == 2 -> 2
     *     D == 3 -> 3
     */
    template<typename STENCIL, int X, int Y, int Z>
    class OffsetHelper;

    template<int DIM, int X, int Y, int Z>
    class OffsetHelper<Moore<DIM, 1>, X, Y, Z>
    {
    public:
        static const int VALUE = 1 * X + 3 * Y + 9 * Z + Sum<Moore, DIM - 1, 1>::VALUE;
    };

    template<int DIM, int X, int Y, int Z>
    class OffsetHelper<VonNeumann<DIM, 1>, X, Y, Z>
    {
    public:
        static const int VALUE = 1 * X + 2 * Y + 3 * Z + DIM;
    };

private:
    template<int BASE, int EXP>
    class Power 
    {
    public:
        static const int VALUE = BASE * Power<BASE, EXP - 1>::VALUE;
    };

    template<int BASE>
    class Power<BASE, 0> 
    {
    public:
        static const int VALUE = 1;
    };

    template<template<int, int> class ADDEND, int INDEX, int RADIUS>
    class Sum
    {
    public:
        static const int VALUE = Sum<ADDEND, INDEX - 1, RADIUS>::VALUE + ADDEND<INDEX, RADIUS>::VOLUME;
    };

    template<template<int, int> class ADDEND, int RADIUS>
    class Sum<ADDEND, 0, RADIUS>
    {
    public:
        static const int VALUE = ADDEND<0, RADIUS>::VOLUME;
    };

    template<int RADIUS, int DIM>
    class VonNeumannHelper
    {
    public:
        static const int VOLUME = VonNeumann<DIM, RADIUS>::VOLUME;
    };

    template<int DIM>
    class VonNeumannHelper<0, DIM>
    {
    public:
        static const int VOLUME = 1;
    };
};

}

#endif
