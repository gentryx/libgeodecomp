#ifndef LIBGEODECOMP_GEOMETRY_STENCILS_H
#define LIBGEODECOMP_GEOMETRY_STENCILS_H

#include <libgeodecomp/geometry/coord.h>

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

    /**
     * Computes \f$\Sigma_{i=0}^{INDEX} ADDEND<i, PARAM2>::VOLUME\f$
     */
    template<template<int, int> class ADDEND, int INDEX, int PARAM2>
    class Sum1;

    /**
     * Computes \f$\Sigma_{i=0}^{INDEX} ADDEND<PARAM1, i>::VOLUME\f$
     */
    template<template<int, int> class ADDEND, int INDEX, int PARAM1>
    class Sum2;

    template<int DIM, int RADIUS, int Z_OFFSET>
    class VonNeumannDimDelta;

public:
    // Utility class to ease enumeration of statements, e.g. copying
    // of all coords in a stencil. BOOST_PP_REPEAT and friends can't
    // save us here as they can't handle the case where the number of
    // iterations is just available as a static const int member of a
    // class. This is mandatory for us -- each stencil has a different
    // volume.
    template<int NUM, template<class C, int I> class TEMPLATE, class PARAM>
    class Repeat
    {
    public:
        void operator()() const
        {
            Repeat<NUM - 1, TEMPLATE, PARAM>()();
            TEMPLATE<PARAM, NUM - 1>()();
        }

        template<typename CARGO>
        void operator()(CARGO cargo) const
        {
            Repeat<NUM - 1, TEMPLATE, PARAM>()(cargo);
            TEMPLATE<PARAM, NUM - 1>()(cargo);
        }

        template<typename CARGO1, typename CARGO2>
        void operator()(CARGO1 cargo1, CARGO2 cargo2) const
        {
            Repeat<NUM - 1, TEMPLATE, PARAM>()(cargo1, cargo2);
            TEMPLATE<PARAM, NUM - 1>()(cargo1, cargo2);
        }

        template<typename CARGO1, typename CARGO2, typename CARGO3>
        void operator()(CARGO1 cargo1, CARGO2 cargo2, CARGO3 cargo3) const
        {
            Repeat<NUM - 1, TEMPLATE, PARAM>()(cargo1, cargo2, cargo3);
            TEMPLATE<PARAM, NUM - 1>()(cargo1, cargo2, cargo3);
        }
    };

    template<template<class C, int I> class TEMPLATE, class PARAM>
    class Repeat<0, TEMPLATE, PARAM>
    {
    public:
        void operator()() const
        {}

        template<typename CARGO>
        void operator()(const CARGO& /* cargo */) const
        {}

        template<typename CARGO1, typename CARGO2>
        void operator()(const CARGO1& /* cargo1 */, const CARGO2& /* cargo2 */) const
        {}

        template<typename CARGO1, typename CARGO2, typename CARGO3>
        void operator()(const CARGO1& /* cargo1 */, const CARGO2& /* cargo2 */, const CARGO3& /* cargo3 */) const
        {}
    };

    /**
     * Same as Repeat, but enabled for CUDA.
     */
    template<int NUM, template<class C, int I> class TEMPLATE, class PARAM>
    class RepeatCuda
    {
    public:
        __host__ __device__
        void operator()() const
        {
            RepeatCuda<NUM - 1, TEMPLATE, PARAM>()();
            TEMPLATE<PARAM, NUM - 1>()();
        }

        template<typename CARGO>
        __host__ __device__
        void operator()(CARGO cargo) const
        {
            RepeatCuda<NUM - 1, TEMPLATE, PARAM>()(cargo);
            TEMPLATE<PARAM, NUM - 1>()(cargo);
        }

        template<typename CARGO1, typename CARGO2>
        __host__ __device__
        void operator()(CARGO1 cargo1, CARGO2 cargo2) const
        {
            RepeatCuda<NUM - 1, TEMPLATE, PARAM>()(cargo1, cargo2);
            TEMPLATE<PARAM, NUM - 1>()(cargo1, cargo2);
        }

        template<typename CARGO1, typename CARGO2, typename CARGO3>
        __host__ __device__
        void operator()(CARGO1 cargo1, CARGO2 cargo2, CARGO3 cargo3) const
        {
            RepeatCuda<NUM - 1, TEMPLATE, PARAM>()(cargo1, cargo2, cargo3);
            TEMPLATE<PARAM, NUM - 1>()(cargo1, cargo2, cargo3);
        }
    };

    template<template<class C, int I> class TEMPLATE, class PARAM>
    class RepeatCuda<0, TEMPLATE, PARAM>
    {
    public:
        __host__ __device__
        void operator()() const
        {}

        template<typename CARGO>
        __host__ __device__
        void operator()(const CARGO& cargo) const
        {}

        template<typename CARGO1, typename CARGO2>
        __host__ __device__
        void operator()(CARGO1 cargo1, CARGO2 cargo2) const
        {}

        template<typename CARGO1, typename CARGO2, typename CARGO3>
        __host__ __device__
        void operator()(CARGO1 cargo1, CARGO2 cargo2, CARGO3 cargo3) const
        {}
    };

    /**
     * The classic Moore neighborhood contains all cells whose spatial
     * distance to the orign (i.e. the current cell) -- as measured by
     * the maximum norm -- is smaller or equal to RADIUS.
     */
    template<int DIMENSIONS, int RAD>
    class Moore
    {
    public:
        static const int RADIUS = RAD;
        static const int DIM = DIMENSIONS;
        static const int VOLUME = Power<RADIUS * 2 + 1, DIM>::VALUE;

        // a list of Classes that derive from FixedCoord and define the stencil's shape
        template<int INDEX>
        class Coords;
    };

    template<int DIMENSIONS, int RADIUS>
    class VonNeumann;

    /**
     * This helper class exists only to work around a bug in GCC 4.4
     * with certain recursive templates.
     */
    template<int DIMENSIONS, int RADIUS>
    class VonNeumannHelper
    {
    public:
        static const int VOLUME = VonNeumann<DIMENSIONS, RADIUS>::VOLUME;
    };

    /**
     * The VonNeumann neighborhood is probably as well known as the
     * Moore neighborhood, but most commonly only used with a RADIUS
     * equal to 1. It replaces the maximum norm with the Manhattan
     * distance.
     */
    template<int DIMENSIONS, int RAD>
    class VonNeumann
    {
    public:
        static const int RADIUS = RAD;
        static const int DIM = DIMENSIONS;
        static const int VOLUME =
            VonNeumann<DIM - 1, RADIUS>::VOLUME +
            2 * Sum2<VonNeumannHelper, DIM - 1, RADIUS>::VALUE -
            2 * VonNeumann<DIM - 1, RADIUS>::VOLUME;

        // a list of Classes that derive from FixedCoord and define the stencil's shape
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
    template<int DIMENSIONS, int RAD>
    class Cross
    {
    public:
        static const int RADIUS = RAD;
        static const int DIM = DIMENSIONS;
        static const int VOLUME = 1 + 2 * RADIUS * DIM;

        // a list of Classes that derive from FixedCoord and define the stencil's shape
        template<int INDEX>
        class Coords;
    };

    /**
     * This is a utility class to aid in adressing all neighboring
     * cells which are packed in a linear array. It's pratically the
     * opposite of the Stencils member types Coords. Examples:
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

    template<int DIM, int RADIUS, int X, int Y, int Z>
    class OffsetHelper<Moore<DIM, RADIUS>, X, Y, Z>
    {
    public:
        static const int VALUE =
            Power<2 * RADIUS + 1, 0>::VALUE * X +
            Power<2 * RADIUS + 1, 1>::VALUE * Y +
            Power<2 * RADIUS + 1, 2>::VALUE * Z +
            RADIUS * Sum1<Moore, DIM - 1, RADIUS>::VALUE;
    };

    template<int DIM, int RADIUS, int X, int Y, int Z>
    class OffsetHelper<VonNeumann<DIM, RADIUS>, X, Y, Z>
    {
    public:
        static const int VALUE =
            1 * X +
            VonNeumannDimDelta<2, RADIUS, Y>::VALUE +
            VonNeumannDimDelta<3, RADIUS, Z>::VALUE +
            (VonNeumann<DIM, RADIUS>::VOLUME - 1) / 2;
    };

private:
    /**
     * The VonNeumann stencil's diamond shape is sadly more complex
     * than the Moore stencil. Given a relative coordinate, we need a
     * way to calculate the index of the corresponding pointer in the
     * pointer array. This class aids by computing the offset in the
     * most significant dimension (e.g. the Z dimension for 3D or the
     * Y dimension for 2D).
     */
    template<int DIM, int RADIUS, int Z_OFFSET>
    class VonNeumannDimDelta
    {
    private:
        static const int DELTA1 =
            Sum2<VonNeumann, DIM - 1, RADIUS>::VALUE -
            (VonNeumann<DIM - 1, RADIUS>::VOLUME - 1) / 2;
        static const int DELTA2 =
            Sum2<VonNeumann, DIM - 1, RADIUS - Z_OFFSET>::VALUE -
            (VonNeumann<DIM - 1, RADIUS - Z_OFFSET>::VOLUME - 1) / 2;
    public:
        static const int VALUE = DELTA1 - DELTA2;
    };

    // Problem: we need to distinguish between positive and negative
    // offsets. Is there no flexible solution for this?
    template<int DIM, int RADIUS>
    class VonNeumannDimDelta<DIM, RADIUS, -1>
    {
    public:
        static const int VALUE = -VonNeumannDimDelta<DIM, RADIUS, 1>::VALUE;
    };

    template<int DIM, int RADIUS>
    class VonNeumannDimDelta<DIM, RADIUS, -2>
    {
    public:
        static const int VALUE = -VonNeumannDimDelta<DIM, RADIUS, 2>::VALUE;
    };

    template<int DIM, int RADIUS>
    class VonNeumannDimDelta<DIM, RADIUS, -3>
    {
    public:
        static const int VALUE = -VonNeumannDimDelta<DIM, RADIUS, 3>::VALUE;
    };

    template<int DIM, int RADIUS>
    class VonNeumannDimDelta<DIM, RADIUS, -4>
    {
    public:
        static const int VALUE = -VonNeumannDimDelta<DIM, RADIUS, 4>::VALUE;
    };

    template<int DIM, int RADIUS>
    class VonNeumannDimDelta<DIM, RADIUS, -5>
    {
    public:
        static const int VALUE = -VonNeumannDimDelta<DIM, RADIUS, 5>::VALUE;
    };

    template<int DIM, int RADIUS>
    class VonNeumannDimDelta<DIM, RADIUS, -6>
    {
    public:
        static const int VALUE = -VonNeumannDimDelta<DIM, RADIUS, 6>::VALUE;
    };

    template<int DIM, int RADIUS>
    class VonNeumannDimDelta<DIM, RADIUS, -7>
    {
    public:
        static const int VALUE = -VonNeumannDimDelta<DIM, RADIUS, 7>::VALUE;
    };

    template<int DIM, int RADIUS>
    class VonNeumannDimDelta<DIM, RADIUS, -8>
    {
    public:
        static const int VALUE = -VonNeumannDimDelta<DIM, RADIUS, 8>::VALUE;
    };

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

    template<template<int, int> class ADDEND, int INDEX, int PARAM2>
    class Sum1
    {
    public:
        static const int VALUE =
            Sum1<ADDEND, INDEX - 1, PARAM2>::VALUE +
            ADDEND<INDEX, PARAM2>::VOLUME;
    };

    template<template<int, int> class ADDEND, int PARAM2>
    class Sum1<ADDEND, 0, PARAM2>
    {
    public:
        static const int VALUE = ADDEND<0, PARAM2>::VOLUME;
    };

    template<template<int, int> class ADDEND, int PARAM1, int INDEX>
    class Sum2
    {
    public:
        static const int VALUE =
            Sum2<ADDEND, PARAM1, INDEX - 1>::VALUE +
            ADDEND<PARAM1, INDEX>::VOLUME;
    };

    template<template<int, int> class ADDEND, int PARAM1>
    class Sum2<ADDEND, PARAM1, 0>
    {
    public:
        static const int VALUE = ADDEND<PARAM1, 0>::VOLUME;
    };
};

#define ADD_COORD(STENCIL, DIM, RADIUS, INDEX, X, Y, Z)                 \
    template<>                                                          \
    template<>                                                          \
    class Stencils::STENCIL<DIM, RADIUS>::Coords<INDEX> : public FixedCoord<X, Y, Z> \
    {};

ADD_COORD(Moore, 1, 1, 0, -1, 0, 0);
ADD_COORD(Moore, 1, 1, 1,  0, 0, 0);
ADD_COORD(Moore, 1, 1, 2,  1, 0, 0);

ADD_COORD(Moore, 1, 2, 0, -2, 0, 0);
ADD_COORD(Moore, 1, 2, 1, -1, 0, 0);
ADD_COORD(Moore, 1, 2, 2,  0, 0, 0);
ADD_COORD(Moore, 1, 2, 3,  1, 0, 0);
ADD_COORD(Moore, 1, 2, 4,  2, 0, 0);

ADD_COORD(Moore, 2, 1, 0, -1, -1, 0);
ADD_COORD(Moore, 2, 1, 1,  0, -1, 0);
ADD_COORD(Moore, 2, 1, 2,  1, -1, 0);
ADD_COORD(Moore, 2, 1, 3, -1,  0, 0);
ADD_COORD(Moore, 2, 1, 4,  0,  0, 0);
ADD_COORD(Moore, 2, 1, 5,  1,  0, 0);
ADD_COORD(Moore, 2, 1, 6, -1,  1, 0);
ADD_COORD(Moore, 2, 1, 7,  0,  1, 0);
ADD_COORD(Moore, 2, 1, 8,  1,  1, 0);

ADD_COORD(Moore, 2, 2,  0, -2, -2, 0);
ADD_COORD(Moore, 2, 2,  1, -1, -2, 0);
ADD_COORD(Moore, 2, 2,  2,  0, -2, 0);
ADD_COORD(Moore, 2, 2,  3,  1, -2, 0);
ADD_COORD(Moore, 2, 2,  4,  2, -2, 0);
ADD_COORD(Moore, 2, 2,  5, -2, -1, 0);
ADD_COORD(Moore, 2, 2,  6, -1, -1, 0);
ADD_COORD(Moore, 2, 2,  7,  0, -1, 0);
ADD_COORD(Moore, 2, 2,  8,  1, -1, 0);
ADD_COORD(Moore, 2, 2,  9,  2, -1, 0);
ADD_COORD(Moore, 2, 2, 10, -2,  0, 0);
ADD_COORD(Moore, 2, 2, 11, -1,  0, 0);
ADD_COORD(Moore, 2, 2, 12,  0,  0, 0);
ADD_COORD(Moore, 2, 2, 13,  1,  0, 0);
ADD_COORD(Moore, 2, 2, 14,  2,  0, 0);
ADD_COORD(Moore, 2, 2, 15, -2,  1, 0);
ADD_COORD(Moore, 2, 2, 16, -1,  1, 0);
ADD_COORD(Moore, 2, 2, 17,  0,  1, 0);
ADD_COORD(Moore, 2, 2, 18,  1,  1, 0);
ADD_COORD(Moore, 2, 2, 19,  2,  1, 0);
ADD_COORD(Moore, 2, 2, 20, -2,  2, 0);
ADD_COORD(Moore, 2, 2, 21, -1,  2, 0);
ADD_COORD(Moore, 2, 2, 22,  0,  2, 0);
ADD_COORD(Moore, 2, 2, 23,  1,  2, 0);
ADD_COORD(Moore, 2, 2, 24,  2,  2, 0);

ADD_COORD(Moore, 3, 1,  0, -1, -1, -1);
ADD_COORD(Moore, 3, 1,  1,  0, -1, -1);
ADD_COORD(Moore, 3, 1,  2,  1, -1, -1);
ADD_COORD(Moore, 3, 1,  3, -1,  0, -1);
ADD_COORD(Moore, 3, 1,  4,  0,  0, -1);
ADD_COORD(Moore, 3, 1,  5,  1,  0, -1);
ADD_COORD(Moore, 3, 1,  6, -1,  1, -1);
ADD_COORD(Moore, 3, 1,  7,  0,  1, -1);
ADD_COORD(Moore, 3, 1,  8,  1,  1, -1);
ADD_COORD(Moore, 3, 1,  9, -1, -1,  0);
ADD_COORD(Moore, 3, 1, 10,  0, -1,  0);
ADD_COORD(Moore, 3, 1, 11,  1, -1,  0);
ADD_COORD(Moore, 3, 1, 12, -1,  0,  0);
ADD_COORD(Moore, 3, 1, 13,  0,  0,  0);
ADD_COORD(Moore, 3, 1, 14,  1,  0,  0);
ADD_COORD(Moore, 3, 1, 15, -1,  1,  0);
ADD_COORD(Moore, 3, 1, 16,  0,  1,  0);
ADD_COORD(Moore, 3, 1, 17,  1,  1,  0);
ADD_COORD(Moore, 3, 1, 18, -1, -1,  1);
ADD_COORD(Moore, 3, 1, 19,  0, -1,  1);
ADD_COORD(Moore, 3, 1, 20,  1, -1,  1);
ADD_COORD(Moore, 3, 1, 21, -1,  0,  1);
ADD_COORD(Moore, 3, 1, 22,  0,  0,  1);
ADD_COORD(Moore, 3, 1, 23,  1,  0,  1);
ADD_COORD(Moore, 3, 1, 24, -1,  1,  1);
ADD_COORD(Moore, 3, 1, 25,  0,  1,  1);
ADD_COORD(Moore, 3, 1, 26,  1,  1,  1);

ADD_COORD(VonNeumann, 1, 1, 0, -1, 0, 0);
ADD_COORD(VonNeumann, 1, 1, 1,  0, 0, 0);
ADD_COORD(VonNeumann, 1, 1, 2,  1, 0, 0);

ADD_COORD(VonNeumann, 1, 2, 0, -2, 0, 0);
ADD_COORD(VonNeumann, 1, 2, 1, -1, 0, 0);
ADD_COORD(VonNeumann, 1, 2, 2,  0, 0, 0);
ADD_COORD(VonNeumann, 1, 2, 3,  1, 0, 0);
ADD_COORD(VonNeumann, 1, 2, 4,  2, 0, 0);

ADD_COORD(VonNeumann, 2, 1, 0,  0, -1, 0);
ADD_COORD(VonNeumann, 2, 1, 1, -1,  0, 0);
ADD_COORD(VonNeumann, 2, 1, 2,  0,  0, 0);
ADD_COORD(VonNeumann, 2, 1, 3,  1,  0, 0);
ADD_COORD(VonNeumann, 2, 1, 4,  0,  1, 0);

ADD_COORD(VonNeumann, 2, 2,  0,  0, -2, 0);
ADD_COORD(VonNeumann, 2, 2,  1, -1, -1, 0);
ADD_COORD(VonNeumann, 2, 2,  2,  0, -1, 0);
ADD_COORD(VonNeumann, 2, 2,  3,  1, -1, 0);
ADD_COORD(VonNeumann, 2, 2,  4, -2,  0, 0);
ADD_COORD(VonNeumann, 2, 2,  5, -1,  0, 0);
ADD_COORD(VonNeumann, 2, 2,  6,  0,  0, 0);
ADD_COORD(VonNeumann, 2, 2,  7,  1,  0, 0);
ADD_COORD(VonNeumann, 2, 2,  9,  2,  0, 0);
ADD_COORD(VonNeumann, 2, 2, 10, -1,  1, 0);
ADD_COORD(VonNeumann, 2, 2, 11,  0,  1, 0);
ADD_COORD(VonNeumann, 2, 2, 12,  1,  1, 0);
ADD_COORD(VonNeumann, 2, 2, 13,  0,  2, 0);

ADD_COORD(VonNeumann, 3, 1, 0,  0,  0, -1);
ADD_COORD(VonNeumann, 3, 1, 1,  0, -1,  0);
ADD_COORD(VonNeumann, 3, 1, 2, -1,  0,  0);
ADD_COORD(VonNeumann, 3, 1, 3,  0,  0,  0);
ADD_COORD(VonNeumann, 3, 1, 4,  1,  0,  0);
ADD_COORD(VonNeumann, 3, 1, 5,  0,  1,  0);
ADD_COORD(VonNeumann, 3, 1, 6,  0,  0,  1);

}

#endif
