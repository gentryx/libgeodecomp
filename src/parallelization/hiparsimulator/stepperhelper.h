#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_stepperhelper_h_
#define _libgeodecomp_parallelization_hiparsimulator_stepperhelper_h_

#include <libgeodecomp/parallelization/hiparsimulator/stepper.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchaccepter.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchprovider.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

/**
 * Convenience class to decouple the Stepper interface from it's
 * internal storage format, which is in turn required by 
 * PatchProvider and PatchAccepter.
 */
template<typename CELL_TYPE, int DIM, typename GRID_TYPE>
class StepperHelper : Stepper<CELL_TYPE, DIM>
{
public:
    enum PatchType {GHOST=0, INNER_SET=1};

    friend class StepperTest;
    typedef GRID_TYPE GridType;
    typedef boost::shared_ptr<PatchProvider<GRID_TYPE> > PatchProviderPtr;
    typedef boost::shared_ptr<PatchAccepter<GRID_TYPE> > PatchAccepterPtr;
    typedef std::deque<PatchProviderPtr> PatchProviderList;
    typedef std::deque<PatchAccepterPtr> PatchAccepterList;
    typedef PartitionManager<
        DIM, typename CELL_TYPE::Topology> MyPartitionManager;

    inline StepperHelper(
        const boost::shared_ptr<MyPartitionManager>& _partitionManager,
        const boost::shared_ptr<Initializer<CELL_TYPE> >& _initializer) :
        Stepper<CELL_TYPE, DIM>(_partitionManager, _initializer)
    {}

    void addPatchProvider(
        const PatchProviderPtr& ghostZonePatchProvider, 
        const PatchType& patchType)
    {
        patchProviders[patchType].push_back(ghostZonePatchProvider);
    }

    void addPatchAccepter(
        const PatchAccepterPtr& ghostZonePatchAccepter, 
        const PatchType& patchType)
    {
        patchAccepters[patchType].push_back(ghostZonePatchAccepter);
    }

    inline virtual const GridType& grid() const =0;

protected:
    PatchProviderList patchProviders[2];
    PatchAccepterList patchAccepters[2];

    inline Initializer<CELL_TYPE>& getInitializer()
    {
        return *this->initializer;
    }

    inline const Initializer<CELL_TYPE>& getInitializer() const
    {
        return *this->initializer;
    }

    inline MyPartitionManager& getPartitionManager()
    {
        return *this->partitionManager;
    }

    inline
    const MyPartitionManager& getPartitionManager() const
    {
        return *this->partitionManager;
    }
};

}
}

#endif
#endif
