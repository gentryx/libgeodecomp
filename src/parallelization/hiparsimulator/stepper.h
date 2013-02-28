#ifndef _libgeodecomp_parallelization_hiparsimulator_stepper_h_
#define _libgeodecomp_parallelization_hiparsimulator_stepper_h_

#include <boost/shared_ptr.hpp>
#include <deque>

#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/parallelization/hiparsimulator/offsethelper.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitionmanager.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchaccepter.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchprovider.h>
#include <libgeodecomp/misc/displacedgrid.h>
#include <libgeodecomp/misc/typetraits.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

/**
 * Abstract interface class. Steppers contain some arbitrary region of
 * the grid which they can update. IO and ghostzone communication are
 * handled via PatchAccepter and PatchProvider objects.
 */
template<typename CELL_TYPE>
class Stepper
{
    friend class StepperTest;
public:
    enum PatchType {GHOST=0, INNER_SET=1};
    typedef typename CELL_TYPE::Topology Topology;
    const static int DIM = Topology::DIMENSIONS;

    typedef DisplacedGrid<CELL_TYPE, Topology, true> GridType;
    typedef PartitionManager<DIM, Topology> MyPartitionManager;
    typedef boost::shared_ptr<PatchProvider<GridType> > PatchProviderPtr;
    typedef boost::shared_ptr<PatchAccepter<GridType> > PatchAccepterPtr;
    typedef std::deque<PatchProviderPtr> PatchProviderList;
    typedef std::deque<PatchAccepterPtr> PatchAccepterList;
    typedef SuperVector<PatchAccepterPtr> PatchAccepterVec;
    typedef SuperVector<PatchProviderPtr> PatchProviderVec;

    inline Stepper(
        const boost::shared_ptr<MyPartitionManager>& _partitionManager,
        Initializer<CELL_TYPE> *_initializer) :
        partitionManager(_partitionManager),
        initializer(_initializer)
    {}

    virtual ~Stepper()
    {}

    virtual void update(int nanoSteps) = 0;

    virtual const GridType& grid() const = 0;

    /**
     * returns current step and nanoStep
     */
    virtual std::pair<int, int> currentStep() const = 0;

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

protected:
    boost::shared_ptr<MyPartitionManager> partitionManager;
    // fixme: replace this by a shared_ptr, refactor all calls to
    // stepper constructors which look like VanillaStepper(... &*init);
    Initializer<CELL_TYPE> *initializer;
    PatchProviderList patchProviders[2];
    PatchAccepterList patchAccepters[2];

    /**
     * calculates a (mostly) suitable offset which (in conjuction with
     * a DisplacedGrid) avoids having grids with a size equal to the
     * whole simulation area on torus topologies.
     */
    inline void guessOffset(Coord<DIM> *offset, Coord<DIM> *dimensions)
    {
        const CoordBox<DIM>& boundingBox = 
            partitionManager->ownRegion().boundingBox();
        OffsetHelper<DIM - 1, DIM, Topology>()(
            offset,
            dimensions,
            boundingBox,
            initializer->gridBox(),
            partitionManager->getGhostZoneWidth());
    }
};

}
}

#endif
