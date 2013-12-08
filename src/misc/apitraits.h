#ifndef LIBGEODECOMP_MISC_APITRAITS_H
#define LIBGEODECOMP_MISC_APITRAITS_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/geometry/stencils.h>
#include <libgeodecomp/geometry/topologies.h>
#include <libgeodecomp/storage/displacedgrid.h>

#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
#include <sstream>
#endif

namespace LibGeoDecomp {

/**
 * APITraits contains a set of classes which can be used to
 * describe/discover the interface between a user-supplied model (cell
 * class) and LibGeoDecomp. More specifically, a cell exports a class
 * named API which derives from certail child classes of APITraits
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
class APITraits
{
public:
    class FalseType
    {};

    class TrueType
    {};

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    // determine whether a cell supports SoA (Struct of Arrays)
    // storage via LibFlatArray.
    template<typename CELL, typename HAS_SOA = void>
    class SelectSoA
    {
    public:
        typedef FalseType Value;
    };

    template<typename CELL>
    class SelectSoA<CELL, typename CELL::API::SupportsSoA>
    {
    public:
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

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    /**
     * Check whether cell has an updateLineX() member.
     */
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

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    /**
     * Does CELL restrict itself to FixedCoord when accessing neighboring cells?
     */
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

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    /**
     * discover which stencil a cell wants to use
     */
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

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    /**
     * Of how many nano steps (intermediate steps) is a whole cell cycle composed?
     */
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

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

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

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    /**
     * determine whether a cell has an architecture-specific speed indicator defined
     */
    template<typename CELL, typename HAS_SPEED = void>
    class SelectSpeedGuide
    {
    public:
        typedef FalseType Value;
    };

    template<typename CELL>
    class SelectSpeedGuide<CELL, typename CELL::API::SupportsSpeed>
    {
    public:
        typedef TrueType Value;
    };

    /**
     * Use this, if you want to use your cell in a heterogenous
     * environment, to specify architecture-specific efficiency hints.
     * This affects the domain decomposition: architectures which are
     * expected to be faster will get a larger initial share of the
     * grid.
     */
    class HasSpeedGuide
    {
    public:
        typedef void SupportsSpeedGuide;
    };

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    /**
     * Decide whether a model can be (de-)serialized with Boost.Serialization.
     */
    template<typename CELL, typename HAS_BOOST_SERIALIZATION = void>
    class SelectBoostSerialization
    {
    public:
        typedef FalseType Value;
    };

    template<typename CELL>
    class SelectBoostSerialization<CELL, typename CELL::API::SupportsBoostSerialization>
    {
    public:
        typedef TrueType Value;
    };

    /**
     * This class can be used to flag cell classes which can be
     * marshalled with Boost.Serialization. This may be advantageous
     * for models which highly diverse memory usage per cell (e.g.
     * n-body codes with heterogeneous particle distributions or AMR
     * codes).
     */
    class HasBoostSerialization
    {
    public:
        typedef void SupportsBoostSerialization;
    };

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    template<typename CELL, typename HAS_SPEED = void>
    class SelectStaticData
    {
    public:
        typedef char Value;
    };

    template<typename CELL>
    class SelectStaticData<CELL, typename CELL::API::SupportsStaticData>
    {
    public:
        typedef typename CELL::API::StaticData Value;
    };


    template<typename STATIC_DATA>
    class HasStaticData
    {
    public:
        typedef void SupportsStaticData;
        typedef STATIC_DATA StaticData;
    };

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    /**
     * This is an n-way switch to allow other classes to select the
     * appropriate type to buffer regions of a grid; for use with GridVecConv.
     */
    template<typename CELL, typename SUPPORTS_SOA = void, typename SUPPORTS_BOOST_SERIALIZATION = void>
    class SelectBufferType
    {
    public:
        typedef std::vector<CELL> Value;

        template<typename REGION>
        static Value create(const REGION& region)
        {
            return Value(region.size());
        }
    };

    template<typename CELL>
    class SelectBufferType<CELL, typename CELL::API::SUPPORTS_SOA, void>
    {
    public:
        typedef std::vector<char> Value;

        template<typename REGION>
        static Value create(const REGION& region)
        {
            return Value(sizeof(CELL) * region.size());
        }
    };

    template<typename CELL>
    class SelectBufferType<CELL, void, typename CELL::API::SUPPORTS_BOOST_SERIALIZATION>
    {
    public:
        typedef std::stringstream Value;

        template<typename REGION>
        static Value create(const REGION& region)
        {
            return Value();
        }
    };
};

}

#endif

