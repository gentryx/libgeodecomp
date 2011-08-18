#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_OPENCL

#ifndef _libgeodecomp_parallelization_hiparsimulator_openclstepper_h_
#define _libgeodecomp_parallelization_hiparsimulator_openclstepper_h_

#include <CL/cl.h>
#include <boost/shared_ptr.hpp>

#include <libgeodecomp/misc/displacedgrid.h>
#include <libgeodecomp/parallelization/hiparsimulator/stepperhelper.h>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<typename CELL_TYPE>
class OpenCLStepper : public StepperHelper<
    DisplacedGrid<CELL_TYPE, typename CELL_TYPE::Topology, false> >
{
public:
    const static int DIM = CELL_TYPE::Topology::DIMENSIONS;
    friend class OpenCLStepperTest;
    typedef DisplacedGrid<
        CELL_TYPE, typename CELL_TYPE::Topology, false> GridType;
    typedef class StepperHelper<GridType> ParentType;
    typedef PartitionManager< 
        DIM, typename CELL_TYPE::Topology> MyPartitionManager;

    inline OpenCLStepper(
        boost::shared_ptr<MyPartitionManager> _partitionManager,
        boost::shared_ptr<Initializer<CELL_TYPE> > _initializer) :
        ParentType(_partitionManager, _initializer)
    {
        // fixme: select OpenCL device in constructor
        curStep = initializer().startStep();
        curNanoStep = 0;
        initGrids();
    }

    inline virtual std::pair<int, int> currentStep() const
    {
        return std::make_pair(curStep, curNanoStep);
    }

    inline virtual void update(int nanoSteps) 
    {
        // fixme: implement me (later)
    }

    inline virtual const GridType& grid() const
    {
        /**
         * fixme:
         * - copy grid from device to hostGrid
         */
        return *hostGrid;
    }


private:
    int curStep;
    int curNanoStep;
    boost::shared_ptr<GridType> hostGrid;

    inline void initGrids()
    {
        const CoordBox<DIM>& gridBox = 
            partitionManager().ownRegion().boundingBox();
        hostGrid.reset(new GridType(gridBox, CELL_TYPE()));
        initializer().grid(&*hostGrid);
        
        /**
         * fixme:
         * - allocate grid on OpenCL device
         * - copy hostGrid to device
         */
    }

    inline MyPartitionManager& partitionManager() 
    {
        return this->getPartitionManager();
    }

    inline const MyPartitionManager& partitionManager() const
    {
        return this->getPartitionManager();
    }

    inline Initializer<CELL_TYPE>& initializer() 
    {
        return this->getInitializer();
    }

    inline const Initializer<CELL_TYPE>& initializer() const
    {
        return this->getInitializer();
    }
};

}
}

#endif
#endif
