#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_hiparsimulator_stepper_h_
#define _libgeodecomp_parallelization_hiparsimulator_stepper_h_

#include <boost/shared_ptr.hpp>
#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitionmanager.h>
#include <libgeodecomp/misc/typetraits.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

/**
 * Abstract interface class. Steppers contain some arbitrary region of
 * the grid which they can update. See StepperHelper for details on
 * accessing ghost zones. Do not inherit directly from Stepper, but
 * rather from StepperHelper.
 *
 * fixme: doxygen syntax to link to StepperHelper...
 */
template<typename CELL_TYPE>
class Stepper
{
public:
    const static int DIM = CELL_TYPE::Topology::DIMENSIONS;

    friend class StepperTest;
    typedef Grid<CELL_TYPE, typename CELL_TYPE::Topology> GridType;
    typedef PartitionManager<DIM, typename CELL_TYPE::Topology> MyPartitionManager;

    inline Stepper(
        const boost::shared_ptr<MyPartitionManager>& _partitionManager,
        const boost::shared_ptr<Initializer<CELL_TYPE> >& _initializer) :
        partitionManager(_partitionManager),
        initializer(_initializer)
    {}

    inline virtual void update(int nanoSteps) = 0;
    //fixme:
    // inline virtual const GridType& grid() const = 0;
    // returns current step and nanoStep
    inline virtual std::pair<int, int> currentStep() const = 0;

protected:
    boost::shared_ptr<MyPartitionManager> partitionManager;
    boost::shared_ptr<Initializer<CELL_TYPE> > initializer;
};

}
}

#endif
#endif
