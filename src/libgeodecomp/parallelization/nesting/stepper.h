#ifndef LIBGEODECOMP_PARALLELIZATION_NESTING_STEPPER_H
#define LIBGEODECOMP_PARALLELIZATION_NESTING_STEPPER_H

#include <libgeodecomp/geometry/partitionmanager.h>
#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/misc/sharedptr.h>
#include <libgeodecomp/parallelization/nesting/offsethelper.h>
#include <libgeodecomp/storage/gridtypeselector.h>
#include <libgeodecomp/storage/patchaccepter.h>
#include <libgeodecomp/storage/patchprovider.h>

// Kill warning 4514 in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <deque>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

/**
 * Abstract interface class. Steppers contain some arbitrary region of
 * the grid which they can update. IO and ghostzone communication are
 * handled via PatchAccepter and PatchProvider objects.
 */
template<typename CELL_TYPE>
class Stepper
{
public:
    friend class StepperTest;

    enum PatchType {GHOST_PHASE_0=0, GHOST_PHASE_1=1, INNER_SET=2};
    typedef typename APITraits::SelectTopology<CELL_TYPE>::Value Topology;
    const static int DIM = Topology::DIM;

    typedef typename APITraits::SelectSoA<CELL_TYPE>::Value SupportsSoA;
    typedef typename GridTypeSelector<CELL_TYPE, Topology, true, SupportsSoA>::Value GridType;

    typedef PartitionManager<Topology> PartitionManagerType;
    typedef typename SharedPtr<PatchProvider<GridType> >::Type PatchProviderPtr;
    typedef typename SharedPtr<PatchAccepter<GridType> >::Type PatchAccepterPtr;
    typedef typename SharedPtr<PartitionManager<Topology> >::Type PartitionManagerPtr;
    typedef typename SharedPtr<Initializer<CELL_TYPE> >::Type InitPtr;
    typedef std::deque<PatchProviderPtr> PatchProviderList;
    typedef std::deque<PatchAccepterPtr> PatchAccepterList;
    typedef std::vector<PatchAccepterPtr> PatchAccepterVec;
    typedef std::vector<PatchProviderPtr> PatchProviderVec;

    inline Stepper(
        PartitionManagerPtr partitionManager,
        InitPtr initializer) :
        partitionManager(partitionManager),
        initializer(initializer)
    {}

    virtual ~Stepper()
    {}

    virtual void update(std::size_t nanoSteps) = 0;

    virtual const GridType& grid() const = 0;

    /**
     * returns current step and nanoStep
     */
    virtual std::pair<std::size_t, std::size_t> currentStep() const = 0;

    void addPatchProvider(
        const PatchProviderPtr& patchProvider,
        const PatchType& patchType)
    {
        patchProviders[patchType].push_back(patchProvider);
    }

    void addPatchAccepter(
        const PatchAccepterPtr& patchAccepter,
        const PatchType& patchType)
    {
        patchAccepters[patchType].push_back(patchAccepter);
    }

    const Chronometer& statistics() const
    {
        return chronometer;
    }

protected:
    PartitionManagerPtr partitionManager;
    InitPtr initializer;
    PatchProviderList patchProviders[3];
    PatchAccepterList patchAccepters[3];
    Chronometer chronometer;

    /**
     * calculates a (mostly) suitable offset which (in conjuction with
     * a DisplacedGrid) avoids having grids with a size equal to the
     * whole simulation area on torus topologies.
     */
    inline void guessOffset(Coord<DIM> *offset, Coord<DIM> *dimensions)
    {
        OffsetHelper<DIM - 1, DIM, Topology>()(
            offset,
            dimensions,
            partitionManager->ownRegion(partitionManager->getGhostZoneWidth()),
            initializer->gridBox());
    }
};

}

#endif
