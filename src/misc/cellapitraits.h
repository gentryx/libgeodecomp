#ifndef LIBGEODECOMP_MISC_CELLAPITRAITS_H
#define LIBGEODECOMP_MISC_CELLAPITRAITS_H

#include <libgeodecomp/misc/displacedgrid.h>
#include <libgeodecomp/misc/stencils.h>
#include <libgeodecomp/misc/topologies.h>

namespace LibGeoDecomp {

template<typename CELL,
         typename TOPOLOGY = Topologies::Cube<2>::Topology,
         bool TOPOLOGICALLY_CORRECT = false>
class SoAGrid;

// fixme: rename this this to "class API"
/**
 * CellAPITraits contains a set of classes which can be used to
 * describe/discover the interface between a user-supplied model (cell
 * class) and LibGeoDecomp. More specifically, a cell exports a class
 * named API which derives from certail child classes of CellAPITraits
 * to allow Simulators and the UpdateFunctor to discover its
 * properties (e.g. number of nano steps, stencil shape, signature and
 * flavor of update() functions...).
 *
 * These classes generally come in pairs: for any feature "foo bar"
 * there is a class HasFooBar which can be derived from by a Cell::API
 * class to signal that Cell supports this feature. Internally
 * LibGeoDecomp will use SelectFooBar<Cell>::Value to discover whether
 * Cell is compatible with "foo bar". This two-part mechanism is
 * necessary since we want to
 *
 * - relieve the user from having to explicitly enable/disable each feature in his model,
 * - provide sensible defaults, and
 * - avoid breaking user code whenever a new feature is introduced.
 */
class CellAPITraitsFixme
{
public:
    class FalseType
    {};

    class TrueType
    {};

    // check whether cell has an updateLineX() member
    template<typename CELL, typename HAS_UPDATE_LINE_X = void>
    class SelectUpdateLineX
    {
    public:
        typedef FalseType Value;
    };

    template<typename CELL>
    class SelectUpdateLineX<CELL, typename CELL::API::SupportsUpdateLineX>
    {
    public:
        typedef TrueType Value;
    };

    // does CELL restrict itself to FixedCoord when accessing neighboring cells?
    template<typename CELL, typename HAS_FIXED_COORDS_ONLY_UPDATE = void>
    class SelectFixedCoordsOnlyUpdate
    {
    public:
        typedef FalseType Value;
    };

    template<typename CELL>
    class SelectFixedCoordsOnlyUpdate<CELL, typename CELL::API::SupportsFixedCoordsOnlyUpdate>
    {
    public:
        typedef TrueType Value;
    };

    // discover which stencil a cell wants to use
    template<typename CELL, typename HAS_STENCIL = void>
    class SelectStencil
    {
    public:
        typedef Stencils::Moore<2, 1> Value;
    };

    template<typename CELL>
    class SelectStencil<CELL, typename CELL::API::SupportsStencil>
    {
    public:
        typedef typename CELL::API::Stencil Value;
    };

    // of how many nano steps (intermediate steps) is a whole cell cycle composed?
    template<typename CELL, typename HAS_NANO_STEPS = void>
    class SelectNanoSteps
    {
    public:
        static const unsigned VALUE = 1;
    };

    template<typename CELL>
    class SelectNanoSteps<CELL, typename CELL::API::SupportsNanoSteps>
    {
    public:
        static const unsigned VALUE = CELL::API::NANO_STEPS;
    };

    template<typename CELL, typename HAS_TOPOLOGY = void>
    class SelectTopology
    {
    public:
        typedef Topologies::Cube<2>::Topology Value;
    };

    template<typename CELL>
    class SelectTopology<CELL, typename CELL::API::SupportsTopology>
    {
    public:
        typedef typename CELL::API::Topology Value;
    };

    // deduce a CELL's optimum grid type
    template<typename CELL, typename HAS_SOA = void>
    class SelectGridType
    {
    public:
        typedef DisplacedGrid<CELL, typename SelectTopology<CELL>::Value> Type;
        typedef FalseType Value;
    };

    template<typename CELL>
    class SelectGridType<CELL, typename CELL::API::SupportsSoA>
    {
    public:
        typedef SoAGrid<CELL, typename SelectTopology<CELL>::Value, false> Type;
        typedef TrueType Value;
    };

    /**
     * Use this qualifier in a cell's API to hint that it supports a
     * Struct of Arrays memory layout.
     */
    class HasSoA
    {
    public:
        typedef void SupportsSoA;
    };

    /**
     * This qualifier should be used to flag models which sport a static
     * updateLineX() function, which is expected to update a streak of
     * cells along the X axis.
     */
    class HasUpdateLineX
    {
    public:
        typedef void SupportsUpdateLineX;
    };

    /**
     * Use this if your model can restrict itself to use only FixedCoord
     * to address its neighbors (via the neighborhood object to update()).
     * This allows the library to apply optimizations which will yield
     * dramatic performance increases for most applications.
     */
    class HasFixedCoordsOnlyUpdate
    {
    public:
        typedef void SupportsFixedCoordsOnlyUpdate;
    };

    /**
     * Allows cells to override the default stencil shape/radius.
     */
    template<typename STENCIL>
    class HasStencil
    {
    public:
        typedef void SupportsStencil;

        typedef STENCIL Stencil;
    };

    /**
     * Defines how many logical time steps constitute one physically
     * correct time step (e.g. LBM kernels often involve two passes:
     * one for streaming, on for collision).
     */
    template<unsigned CELL_NANO_STEPS>
    class HasNanoSteps
    {
    public:
        typedef void SupportsNanoSteps;

        static const unsigned NANO_STEPS = CELL_NANO_STEPS;
    };

    /**
     * Here cells can specify whether they expect a different topology
     * (number of spatial dimensions, type of boundary conditions)
     * than defined in the default.
     */
    template<typename TOPOLOGY>
    class HasTopology
    {
    public:
        typedef void SupportsTopology;

        typedef TOPOLOGY Topology;
    };

    /**
     * Convenience overload to simplify most common topology specifications
     */
    template<int DIM>
    class HasCubeTopology : public HasTopology<typename Topologies::Cube<DIM>::Topology>
    {};

    /**
     * Same as for HasCubeTopology: overload purely for convenience.
     */
    template<int DIM>
    class HasTorusTopology : public HasTopology<typename Topologies::Torus<DIM>::Topology>
    {};
};

// fixme: kill this!
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

