#ifndef LIBGEODECOMP_MISC_APITRAITS_H
#define LIBGEODECOMP_MISC_APITRAITS_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/geometry/floatcoord.h>
#include <libgeodecomp/geometry/stencils.h>
#include <libgeodecomp/geometry/topologies.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>

#include <libflatarray/detail/disable_system_header_warnings_1.hpp>
#ifdef LIBGEODECOMP_WITH_MPI
#  include <mpi.h>
#endif
#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
#  include <sstream>
#endif
#include <libflatarray/detail/disable_system_header_warnings_2.hpp>

namespace LibGeoDecomp {

// Hardwire this warning to off as MSVC would otherwise complain about
// inline functions not being included in object files:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#ifdef LIBGEODECOMP_WITH_MPI
class Typemaps;
#endif

namespace APITraitsHelpers {

#ifdef LIBGEODECOMP_WITH_MPI

/**
 * Utility class which provides searches a provider class for a given
 * MPI datatype.
 */
template<typename PROVIDER, typename CELL>
class DefaultMPIDataTypeLookup
{
public:
    static inline MPI_Datatype value()
    {
        return PROVIDER::lookup(reinterpret_cast<CELL*>(0));
    }
};

#endif

}

/**
 * APITraits contains a set of classes which can be used to
 * describe/discover the interface between a user-supplied model (cell
 * class) and LibGeoDecomp. Examples of how to use these can be found
 * in src/examples/
 *
 * A cell exports a class named API which derives from certail child
 * classes of APITraits to allow Simulators and the UpdateFunctor to
 * discover its properties (e.g. number of nano steps, stencil shape,
 * signature and flavor of update() functions...).
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
    class TrueType;

    class FalseType
    {
    public:
        inline operator bool() const
        {
            return false;
        }

        TrueType operator!() const
        {
            return TrueType();
        }
    };

    class TrueType
    {
    public:
        inline operator bool() const
        {
            return true;
        }

        FalseType operator!() const
        {
            return FalseType();
        }
    };

    /**
     * Helper class for specializations. Thanks to Hartmut Kaiser for
     * coming up with this.
     */
    template<typename T>
    class AlwaysVoid
    {
    public:
        typedef void Type;
    };

    /**
     * Type trait which allows external code to detect if a class has
     * a member function value(). Gratefully based off
     * http://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Member_Detector
     */
    template<typename T, typename RESULT, typename PARAM>
    class HasLookupMemberFunction
    {
    private:
        // to detect member functions, the second parameter would have
        // to be changed to "RESULT (U::*)()"
        template <typename U, RESULT (*)(PARAM*)> struct Check;
        template <typename U> static char func(Check<U, &U::lookup> *);
        template <typename U> static int func(...);
    public:
        enum { value = sizeof(func<T>(0)) == sizeof(char) };
    };

    /**
     * Type trait which allows external code to detect if a class has
     * a member function value(). Gratefully based off
     * http://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Member_Detector
     */
    template<typename T, typename RESULT>
    class HasValueFunction
    {
    private:
        template <typename U, RESULT (*)()> struct Check;
        template <typename U> static char func(Check<U, &U::value> *);
        template <typename U> static int func(...);
    public:
        enum { value = sizeof(func<T>(0)) == sizeof(char) };
    };

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    template<typename CELL, typename HAS_API = void>
    class SelectAPI
    {
    public:
        class Value
        {};
    };

    /**
     * Allows library code to pull in a model's whole API without
     * requiring the user to specify one at all (if none is defined,
     * we'll return an empty class).
     */
    template<typename CELL>
    class SelectAPI<CELL, typename AlwaysVoid<typename CELL::API>::Type>
    {
    public:
        typedef typename CELL::API Value;
    };

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

    /**
     * Enforces that an unstructured grid is to be used for the
     * simulation. Not to be confused with HasUnstructuredGrid, which
     * is a trait for IO.
     */
    class HasUnstructuredTopology: public HasTopology<Topologies::Unstructured::Topology>
    {};

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    template<typename CELL, typename HAS_SELL_C = void>
    class SelectSellC
    {
    public:
        static const unsigned VALUE = 4;
    };

    template<typename CELL>
    class SelectSellC<CELL, typename CELL::API::SupportsSellC>
    {
    public:
        static const int VALUE = CELL::API::SELL_C;
    };

    /**
     * For unstructured grid, this specifies the C of SELL-C-q format. Default: 4
     * which is suitable for AVX.
     */
    template<int C>
    class HasSellC
    {
    public:
        typedef void SupportsSellC;

        static const int SELL_C = C;
    };

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    template<typename CELL, typename HAS_SELL_SIGMA = void>
    class SelectSellSigma
    {
    public:
        static const int VALUE = 1;
    };

    template<typename CELL>
    class SelectSellSigma<CELL, typename CELL::API::SupportsSellSigma>
    {
    public:
        static const int VALUE = CELL::API::SELL_SIGMA;
    };

    /**
     * For unstructured grid, this specifies the SIGMA of SELL-C-q format.
     * Should be C^2. Default is 1 which means nothing is sorted.
     */
    template<int SIGMA>
    class HasSellSigma
    {
    public:
        typedef void SupportsSellSigma;

        static const int SELL_SIGMA = SIGMA;
    };

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    template<typename CELL, typename HAS_SELL_MATRICES = void>
    class SelectSellMatrices
    {
    public:
        static const std::size_t VALUE = 1;
    };

    template<typename CELL>
    class SelectSellMatrices<CELL, typename CELL::API::SupportsSellMatrices>
    {
    public:
        static const std::size_t VALUE = CELL::API::SELL_MATRICES;
    };

    /**
     * For unstructured grid, this specifies the amount of different matrices.
     * Default is 1.
     */
    template<std::size_t MATRICES>
    class HasSellMatrices
    {
    public:
        typedef void SupportsSellMatrices;

        static const std::size_t SELL_MATRICES = MATRICES;
    };

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    template<typename CELL, typename HAS_SELL_TYPE = void>
    class SelectSellType
    {
    public:
        typedef double Value;
    };

    template<typename CELL>
    class SelectSellType<CELL, typename CELL::API::SupportsSellType>
    {
    public:
        typedef typename CELL::API::SellType Value;
    };

    /**
     * For unstructured grid, this specifies the type of values inside the
     * SELL-C-q matrices. Default is double.
     */
    template<typename SELL_TYPE>
    class HasSellType
    {
    public:
        typedef void SupportsSellType;

        typedef SELL_TYPE SellType;
    };

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    /**
     * determine whether a cell has an architecture-specific speed indicator defined
     */
    template<typename CELL, typename HAS_SPEED = void>
    class SelectSpeedGuide
    {
    public:
        static double value()
        {
            return 1.0;
        }
    };

    template<typename CELL>
    class SelectSpeedGuide<CELL, typename CELL::API::SupportsSpeed>
    {
    public:
        static double value()
        {
            return CELL::cellSpeed();
        }
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

    template<typename CELL,
             typename HAS_MPI_DATA_TYPE = void,
             typename MPI_DATA_TYPE_RETRIEVAL = void>
    class SelectMPIDataType
    {
    public:
    };

#ifdef LIBGEODECOMP_WITH_MPI
    template<typename CELL>
    class SelectMPIDataType<
        CELL,
        typename CELL::API::SupportsMPIDataType,
        typename CELL::API::SupportsCustomMPIDataType>
    {
    public:
        static inline MPI_Datatype value()
        {
            return CELL::API::getMPIDataType();
        }
    };

    template<typename CELL>
    class SelectMPIDataType<
        CELL,
        typename CELL::API::SupportsMPIDataType,
        typename CELL::API::SupportsAutoGeneratedMPIDataType>
    {
    public:
        static inline MPI_Datatype value()
        {
            return APITraitsHelpers::
                DefaultMPIDataTypeLookup<
                typename CELL::API::MPIDataTypeProvider,
                CELL>::value();
        }
    };

    template<typename CELL>
    class SelectMPIDataType<
        CELL,
        typename CELL::API::SupportsMPIDataType,
        typename CELL::API::SupportsPredefinedMPIDataType>
    {
    public:
        static inline MPI_Datatype value()
        {
            return CELL::API::MPIDataTypeProvider::lookup(
                reinterpret_cast<typename CELL::API::MPIDataTypeBase*>(0));
        }
    };

    /**
     * Use this specifier to give LibGeoDecomp access to the MPI data
     * type which can be used to communicate instances of a cell. Not
     * required for SoA codes and models which use special
     * serialization schemes (for instance Boost.Serialization).
     */
    template<typename CELL>
    class HasCustomMPIDataType
    {
    public:
        typedef void SupportsMPIDataType;
        typedef void SupportsCustomMPIDataType;

        static inline MPI_Datatype getMPIDataType()
        {
            return CELL::MPIDataType;
        }
    };
#endif

    /**
     * This specifier indicates that some PROVIDER class can deliver
     * the MPI data type for the cell class via a lookup() function.
     */
    template<class PROVIDER>
    class HasAutoGeneratedMPIDataType
    {
    public:
        typedef void SupportsMPIDataType;
        typedef void SupportsAutoGeneratedMPIDataType;
        typedef PROVIDER MPIDataTypeProvider;
    };

    /**
     * This specifier indicates that some PROVIDER class can deliver
     * the MPI data type for the cell class via a lookup() function.
     */
    template<typename BASE_TYPE>
    class HasPredefinedMPIDataType
    {
    public:
        typedef void SupportsMPIDataType;
        typedef void SupportsPredefinedMPIDataType;
#ifdef LIBGEODECOMP_WITH_MPI
        typedef Typemaps MPIDataTypeProvider;
#endif
        typedef BASE_TYPE MPIDataTypeBase;
    };

    /**
     * Use this specifyer if your cell is bitwise serializable
     */
    template<typename CELL>
    class HasOpaqueMPIDataType
    {
    public:
        typedef void SupportsMPIDataType;
        typedef void SupportsCustomMPIDataType;

#ifdef LIBGEODECOMP_WITH_MPI
        static inline MPI_Datatype getMPIDataType()
        {
            static MPI_Datatype datatype = MPI_DATATYPE_NULL;
            if (datatype == MPI_DATATYPE_NULL) {
                MPI_Type_contiguous(sizeof(CELL), MPI_CHAR, &datatype);
                MPI_Type_commit(&datatype);
            }

            return datatype;
        }
#endif
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

    /**
     * Some models have a need for static, but mutable data blocks,
     * e.g. coefficients which need to be controlled by steerers. This
     * class allows users to appropriately flag their classes so that
     * LibGeoDecomp can take over handling of these. Handling refers
     * to synchronizing them on a GPU and between time steps.
     */
    template<typename STATIC_DATA>
    class HasStaticData
    {
    public:
        typedef void SupportsStaticData;
        typedef STATIC_DATA StaticData;
    };

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    template<typename CELL, typename HAS_THREADED_UPDATE = void>
    class SelectThreadedUpdate
    {
    public:
        class Value
        {
        public:
            inline FalseType hasOpenMP() const
            {
                return FalseType();
            }

            inline FalseType hasHPX() const
            {
                return FalseType();
            }

            inline FalseType hasCUDA() const
            {
                return FalseType();
            }

            inline int granularity() const
            {
                return 16384;
            }
        };
    };

    template<typename CELL>
    class SelectThreadedUpdate<CELL, typename CELL::API::SupportsThreadedUpdate>
    {
    public:
        class Value
        {
        public:
            typedef typename CELL::API::HasOpenMP HasOpenMP;
            typedef typename CELL::API::HasHPX    HasHPX;
            typedef typename CELL::API::HasCUDA   HasCUDA;

            inline HasOpenMP hasOpenMP() const
            {
                return HasOpenMP();
            }

            inline HasHPX hasHPX() const
            {
                return HasHPX();
            }

            inline HasCUDA hasCUDA() const
            {
                return HasCUDA();
            }

            inline int granularity() const
            {
                return CELL::API::GRANULARITY_VALUE;
            }
        };
    };

    /**
     * This trait gives models control over the threading strategy
     * taken inside LibGeoDecomp. The GRANULARITY may be used to
     * suggest LibGeoDecomp to give smaller heaps of work to each
     * thread. This can improve scalability when the amount of Streaks
     * is low compared to the number of cores.
     *
     * Some models (e.g. n-body codes) are so compute intensive, that
     * it may be advisable to use multiple threads for updating a
     * single cell -- as opposed to using multiple threads for dijunct
     * regions of the grid. This trait can be used to tell
     * LibGeoDecomp which threading strategy should be used. The model
     * is responsible to provide a suitable implementation (e.g. based
     * on OpenMP or HPX). On CUDA each cell will get its own thread
     * block.
     */
    template<int GRANULARITY, typename HAS_OPENMP = FalseType, typename HAS_HPX = FalseType, typename HAS_CUDA = FalseType>
    class HasThreadedUpdate
    {
    public:
        typedef void SupportsThreadedUpdate;

        typedef HAS_OPENMP HasOpenMP;
        typedef HAS_HPX    HasHPX;
        typedef HAS_CUDA   HasCUDA;

        static const int GRANULARITY_VALUE = GRANULARITY;
    };

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    template<typename CELL, typename HAS_SEPARATE_CUDA_UPDATE = void>
    class SelectSeparateCUDAUpdate
    {
    public:
        typedef FalseType Value;
    };

    template<typename CELL>
    class SelectSeparateCUDAUpdate<CELL, typename CELL::API::SupportsSeparateCUDAUpdate>
    {
    public:
        typedef TrueType Value;
    };

    /**
     * Sometime cells may need to roll different code on CUDA than on
     * the CPU. This trait will make the update functor call
     * updateCUDA() instead of update() when running on the GPU.
     */
    class HasSeparateCUDAUpdate
    {
    public:
        typedef void SupportsSeparateCUDAUpdate;
    };

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    template<typename CELL, typename HAS_COORD_TYPE = void>
    class SelectCoordType
    {
    public:
        typedef FloatCoord<2> Value;
    };

    template<typename CELL>
    class SelectCoordType<CELL, typename CELL::API::SupportsCoordType>
    {
    public:
        typedef typename CELL::API::CoordType Value;
    };

    /**
     * This trait is used by unstructured grid and meshfree codes to
     * set the coordinate type by which the elements are represemted.
     * The typename needs to match the dimensions of the model's
     * topology.
     */
    template<typename COORD>
    class HasCoordType
    {
    public:
        typedef COORD CoordType;

        typedef void SupportsCoordType;
    };

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    template<typename CELL, typename HAS_ID_TYPE = void>
    class SelectIDType
    {
    public:
        typedef int Value;
    };

    template<typename CELL>
    class SelectIDType<CELL, typename CELL::API::SupportsIDType>
    {
    public:
        typedef typename CELL::API::IDType Value;
    };

    /**
     * Like HasCoordType, this trait can be used to set the type which
     * is used in meshfree and unstructured grid codes to uniquely
     * identify elements.
     */
    template<typename ID>
    class HasIDType
    {
    public:
        typedef ID IDType;

        typedef void SupportsIDType;
    };

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    template<typename CELL, typename HAS_POINT_MESH = void>
    class SelectPointMesh
    {
    public:
        typedef FalseType Value;
    };

    template<typename CELL>
    class SelectPointMesh<CELL, typename CELL::API::SupportsPointMesh>
    {
    public:
        typedef TrueType Value;
    };

    /**
     * Indicates that the model has particles or features other
     * entities that can be represented by a point mesh. As an
     * example, the SiloWriter can then dump that mesh to an archive.
     * See the Voronoi example for instructions on how to use this
     * feature.
     */
    class HasPointMesh
    {
    public:
        typedef void SupportsPointMesh;
    };

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    /**
     * All stencil codes are based on a regular grid, but even
     * particle-in-cell codes may use such a grid to organizing the
     * particles and computing certain variables (e.g. direction and
     * magnitude of the field).
     *
     * By default plugins like the SiloWriter will output such a grid,
     * but this behavior can be surpressed by a model via the
     * HasNoRegularGrid trait.
     */
    template<typename CELL, typename HAS_REGULAR_GRID = void>
    class SelectRegularGrid
    {
    public:
        typedef TrueType Value;
        static const int DIM = SelectTopology<CELL>::Value::DIM;

        static inline void value(
            FloatCoord<DIM> *quadrantDim,
            FloatCoord<DIM> *origin,
            std::vector<std::string> *axisUnits)
        {
            *quadrantDim = FloatCoord<DIM>::diagonal(1.0);
            *origin      = FloatCoord<DIM>::diagonal(0.0);

            axisUnits->clear();
            for (int i = 0; i < DIM; ++i) {
                *axisUnits << std::string("");
            }
        }

        static inline void value(
            FloatCoord<SelectTopology<CELL>::Value::DIM> *quadrantDim,
            FloatCoord<SelectTopology<CELL>::Value::DIM> *origin)
        {
            std::vector<std::string> ignoredUnits;
            value(quadrantDim, origin, &ignoredUnits);
        }
    };

    template<typename CELL>
    class SelectRegularGrid<CELL, typename CELL::API::SupportsCustomRegularGrid>
    {
    public:
        typedef TrueType Value;
        static const int DIM = SelectTopology<CELL>::Value::DIM;

        static inline void value(
            FloatCoord<DIM> *quadrantDim,
            FloatCoord<DIM> *origin,
            std::vector<std::string> *axisUnits)
        {
            *quadrantDim = typename CELL::API().getRegularGridSpacing();
            *origin      = typename CELL::API().getRegularGridOrigin();
            *axisUnits   = typename CELL::API().template getRegularGridAxisUnits<DIM>();
        }

        static inline void value(
            FloatCoord<DIM> *quadrantDim,
            FloatCoord<DIM> *origin)
        {
            std::vector<std::string> ignoredUnits;
            value(quadrantDim, origin, &ignoredUnits);
        }
    };

    template<typename CELL>
    class SelectRegularGrid<CELL, typename CELL::API::SupportsNoRegularGrid>
    {
    public:
        typedef FalseType Value;
    };

    /**
     * Use this trait to flag models which contain a regular grid that
     * have a non-stanard spacing.
     *
     * The spatial extent of the cells needs to be configured. See the
     * src/examples/voronoi for an example on how to do this.
     */
    class HasCustomRegularGrid
    {
    public:
        typedef void SupportsCustomRegularGrid;

        // provide a default here as only very few users will want to
        // override it. Some components, e.g. the VisItWriter, will
        // use this info to label plots accordingly.
        template<int DIM>
        std::vector<std::string> getRegularGridAxisUnits()
        {
            std::vector<std::string> axisUnits;

            for (int i = 0; i < DIM; ++i) {
                axisUnits << std::string("");
            }

            return axisUnits;
        }
    };

    class HasNoRegularGrid
    {
    public:
        typedef void SupportsNoRegularGrid;
    };

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    template<typename CELL, typename HAS_UNSTRUCTURED_GRID = void>
    class SelectUnstructuredGrid
    {
    public:
        typedef FalseType Value;
    };

    template<typename CELL>
    class SelectUnstructuredGrid<CELL, typename CELL::API::SupportsUnstructuredGrid>
    {
    public:
        typedef TrueType Value;
    };

    /**
     * This trait can be used by Writers and other IO components to
     * discover that a cell has an unstructured grid whose nodes are
     * stored in its cells (e.g. a regular grid of cells, which act as
     * containers for the unstructured grid). This has no influence on
     * the grid type to be used by the simulator (see
     * HasUnstructuredTopology).
     *
     * The shape of each zone (element) will be represented by a sequence of
     * coordinates. See the Voronoi example for instructions on how to
     * use this feature.
     */
    class HasUnstructuredGrid
    {
    public:
        typedef void SupportsUnstructuredGrid;
    };

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    template<typename CELL, typename HAS_TEMPLATE_NAME = void>
    class SelectPositionChecker
    {
    public:
        template<int DIM>
        inline
        static bool value(
            const CELL& particle,
            const FloatCoord<DIM>& origin,
            const FloatCoord<DIM>& oppositeCorner)
        {
            return
                origin.dominates(particle.getPos()) &&
                particle.getPos().strictlyDominates(oppositeCorner);
        }
    };

    template<typename CELL>
    class SelectPositionChecker<CELL, typename CELL::API::SupportsCustomPositionChecker>
    {
    public:
        /**
         * User code is expected to return true iff a particle's
         * position falls into the cubic domain described by the two
         * coordinates (container dimensions = oppositeCorner - origin).
         */
        template<int DIM>
        inline
        static bool value(
            const CELL& particle,
            const FloatCoord<DIM>& origin,
            const FloatCoord<DIM>& oppositeCorner)
        {
            return CELL::API::checkPosition(particle, origin, oppositeCorner);
        }
    };

    /**
     * This is a hook which allows user code to provide its own
     * position checker. See the classes above for which kind of
     * function the user's API should provide.
     *
     * This is used by container cells such as BoxCell to determine
     * whether a given particle resides within their domain. The
     * edges/faces on the opposite side are considered to belong to
     * the next container.
     *
     * Example:
     * - origin: (10, 20)
     * - dimension: (8, 5)
     * - oppositeCorner: (18, 25)
     *
     * - point (10, 20) -> in
     * - point (10, 22) -> in
     * - point (11, 20) -> in
     * - point (14, 24) -> in
     * - point (18, 21) -> out
     * - point (11, 25) -> out
     * - point (18, 25) -> out
     */
    class HasCustomPositionChecker
    {
    public:
        typedef void SupportsCustomPositionChecker;
    };

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    template<typename CELL, typename HAS_TEMPLATE_NAME = void>
    class SelectMessageType
    {
    public:
        typedef CELL Value;
    };

    template<typename CELL>
    class SelectMessageType<CELL, typename CELL::API::SupportsCustomMessageType>
    {
    public:
        typedef typename CELL::API::MessageType Value;
    };

    /**
     * Use this trait if your simulation model would rather send
     * objects of a specific type to neighbors, rather than
     * synchronizing whole instances of cells. This can, depending on
     * the model, save considerable amounts of communication volume
     * (e.g. DG-SWEM).
     */
    template<typename MESSAGE_TYPE>
    class HasCustomMessageType
    {
    public:
        typedef MESSAGE_TYPE MessageType;
        typedef void SupportsCustomMessageType;
    };

    // XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    // Trait Template:

    // template<typename CELL, typename HAS_TEMPLATE_NAME = void>
    // class SelectTemplateName
    // {
    // public:
    //     typedef FalseType Value;
    // };

    // template<typename CELL>
    // class SelectTemplateName<CELL, typename CELL::API::SupportsTemplateName>
    // {
    // public:
    //     typedef TrueType Value;
    // };

    // /**
    //  * This is just an empty template for adding new traits.
    //  *
    //  */
    // class HasTemplateName
    // {
    // public:
    //     typedef void SupportsTemplateName;
    // };
};

inline bool operator&&(APITraits::TrueType, bool other)
{
    return other;
}

inline bool operator&&(bool other, APITraits::TrueType)
{
    return other;
}

inline bool operator&&(APITraits::FalseType, bool /* other */)
{
    return false;
}

inline bool operator&&(bool /* other */, APITraits::FalseType)
{
    return false;
}

inline APITraits::TrueType operator&&(APITraits::TrueType, APITraits::TrueType)
{
    return APITraits::TrueType();
}

inline APITraits::FalseType operator&&(APITraits::TrueType, APITraits::FalseType)
{
    return APITraits::FalseType();
}

inline APITraits::FalseType operator&&(APITraits::FalseType, APITraits::TrueType)
{
    return APITraits::FalseType();
}

inline APITraits::FalseType operator&&(APITraits::FalseType, APITraits::FalseType)
{
    return APITraits::FalseType();
}

inline APITraits::TrueType operator==(APITraits::TrueType, APITraits::TrueType)
{
    return APITraits::TrueType();
}

inline APITraits::FalseType operator==(APITraits::TrueType, APITraits::FalseType)
{
    return APITraits::FalseType();
}

inline APITraits::FalseType operator==(APITraits::FalseType, APITraits::TrueType)
{
    return APITraits::FalseType();
}

inline APITraits::TrueType operator==(APITraits::FalseType, APITraits::FalseType)
{
    return APITraits::TrueType();
}

inline APITraits::FalseType operator!=(APITraits::TrueType, APITraits::TrueType)
{
    return APITraits::FalseType();
}

inline APITraits::TrueType operator!=(APITraits::TrueType, APITraits::FalseType)
{
    return APITraits::TrueType();
}

inline APITraits::TrueType operator!=(APITraits::FalseType, APITraits::TrueType)
{
    return APITraits::TrueType();
}

inline APITraits::TrueType operator!=(APITraits::FalseType, APITraits::FalseType)
{
    return APITraits::TrueType();
}

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif
