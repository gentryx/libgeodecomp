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
template<typename GRID_TYPE>
class StepperHelper : public Stepper<typename GRID_TYPE::CellType>
{
public:
    friend class StepperTest;
    enum PatchType {GHOST=0, INNER_SET=1};

    typedef GRID_TYPE GridType;
    typedef typename GridType::CellType CellType;
    const static int DIM = CellType::Topology::DIMENSIONS;
    typedef boost::shared_ptr<PatchProvider<GRID_TYPE> > PatchProviderPtr;
    typedef boost::shared_ptr<PatchAccepter<GRID_TYPE> > PatchAccepterPtr;
    typedef std::deque<PatchProviderPtr> PatchProviderList;
    typedef std::deque<PatchAccepterPtr> PatchAccepterList;
    typedef PartitionManager<
        DIM, typename CellType::Topology> MyPartitionManager;

    inline StepperHelper(
        const boost::shared_ptr<MyPartitionManager>& _partitionManager,
        const boost::shared_ptr<Initializer<CellType> >& _initializer) :
        Stepper<CellType>(_partitionManager, _initializer)
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

    inline Initializer<CellType>& getInitializer()
    {
        return *this->initializer;
    }

    inline const Initializer<CellType>& getInitializer() const
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
