#ifndef LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_OPENCLSTEPPER_H
#define LIBGEODECOMP_PARALLELIZATION_HIPARSIMULATOR_OPENCLSTEPPER_H

#include <libgeodecomp/misc/updatefunctor.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchbufferfixed.h>
#include <libgeodecomp/parallelization/hiparsimulator/stepper.h>

#include "openclwrapper.h"

namespace LibGeoDecomp {

  template<typename CELL, typename DATA_TYPE>
  class OpenCLCellInterface {
    public:
      static std::string kernel_file(void) {
        return CELL::kernel_file();
      }
      static std::string kernel_function(void) {
        return CELL::kernel_function();
      }
      static std::string cl_struct_code(void) {
        return CELL::cl_struct_code();
      }
      static size_t sizeof_data(void) {
        return CELL::sizeof_data();
      }
      virtual DATA_TYPE * data(void) = 0;
  };

namespace HiParSimulator {

template<typename CELL_TYPE, typename DATA_TYPE>
class OpenCLStepper : public Stepper<CELL_TYPE>
{
    friend class OpenCLStepperRegionTest;
    friend class OpenCLStepperBasicTest;
    friend class OpenCLStepperTest;
public:
    const static int DIM = CELL_TYPE::Topology::DIM;

    typedef class Stepper<CELL_TYPE> ParentType;
    typedef typename ParentType::GridType GridType;
    typedef PartitionManager<DIM, typename CELL_TYPE::Topology> PartitionManagerType;
    typedef PatchBufferFixed<GridType, GridType, 1> PatchBufferType1;
    typedef PatchBufferFixed<GridType, GridType, 2> PatchBufferType2;
    typedef typename ParentType::PatchAccepterVec PatchAccepterVec;

    using Stepper<CELL_TYPE>::addPatchAccepter;
    using Stepper<CELL_TYPE>::initializer;
    using Stepper<CELL_TYPE>::guessOffset;
    using Stepper<CELL_TYPE>::patchAccepters;
    using Stepper<CELL_TYPE>::patchProviders;
    using Stepper<CELL_TYPE>::partitionManager;

    inline OpenCLStepper(
        unsigned int platform_id, unsigned int device_id,
        boost::shared_ptr<PartitionManagerType> partitionManager,
        boost::shared_ptr<Initializer<CELL_TYPE> > initializer,
        const PatchAccepterVec ghostZonePatchAccepters = PatchAccepterVec(),
        const PatchAccepterVec innerSetPatchAccepters = PatchAccepterVec()) :
        ParentType(partitionManager, initializer),
        platform_id(platform_id), device_id(device_id)
    {
        curStep = initializer->startStep();
        curNanoStep = 0;

        for (int i = 0; i < ghostZonePatchAccepters.size(); ++i) {
            addPatchAccepter(ghostZonePatchAccepters[i], ParentType::GHOST);
        }

        for (int i = 0; i < innerSetPatchAccepters.size(); ++i) {
            addPatchAccepter(innerSetPatchAccepters[i], ParentType::INNER_SET);
        }

        initGrids();

        try {
          std::string kernel_file =
            OpenCLCellInterface<CELL_TYPE, DATA_TYPE>::kernel_file();
          std::string kernel_function =
            OpenCLCellInterface<CELL_TYPE, DATA_TYPE>::kernel_function();
          size_t sizeof_data =
            OpenCLCellInterface<CELL_TYPE, DATA_TYPE>::sizeof_data();

          std::vector<OpenCLWrapper::data_t> data;
          std::vector<OpenCLWrapper::point_t> points;

          auto box = initializer->gridBox();
          for (auto & p : box) {
            auto & cell =
              dynamic_cast<OpenCLCellInterface<CELL_TYPE, DATA_TYPE> &>((*oldGrid)[p]);
            points.push_back(std::make_tuple(p.x(), p.y(), p.z()));
            data.push_back(cell.data());
          }

          int x_size = box.dimensions.x()
            , y_size = box.dimensions.y()
            , z_size = box.dimensions.z();

          oclwrapper = OpenCLWrapper_Ptr(
              new OpenCLWrapper(platform_id, device_id,
                                kernel_file, kernel_function,
                                sizeof_data, x_size, y_size, z_size));

          oclwrapper->loadPoints(points);
          oclwrapper->loadHostData(data);

        } catch (std::exception & e) {
          exit(EXIT_FAILURE);
        }
    }

    inline virtual void update(int nanoSteps)
    {
        for (int i = 0; i < nanoSteps; ++i) {
            update();
        }
    }

    inline virtual std::pair<int, int> currentStep() const
    {
        return std::make_pair(curStep, curNanoStep);
    }

    inline virtual const GridType& grid() const
    {
        return *oldGrid;
    }

private:
    typedef std::shared_ptr<OpenCLWrapper> OpenCLWrapper_Ptr;

    unsigned int platform_id, device_id;

    int curStep;
    int curNanoStep;
    int validGhostZoneWidth;
    boost::shared_ptr<GridType> oldGrid;
    boost::shared_ptr<GridType> newGrid;
    PatchBufferType2 rimBuffer;
    PatchBufferType1 kernelBuffer;
    Region<DIM> kernelFraction;

    OpenCLWrapper_Ptr oclwrapper;

    inline void update()
    {
        unsigned index = ghostZoneWidth() - --validGhostZoneWidth;
        const Region<DIM>& region = partitionManager->innerSet(index);

        oclwrapper->run();
        oclwrapper->finish();

        ++curNanoStep;
        if (curNanoStep == CELL_TYPE::nanoSteps()) {
            curNanoStep = 0;
            curStep++;
        }

        notifyPatchAccepters(region, ParentType::INNER_SET, globalNanoStep());
        notifyPatchProviders(region, ParentType::INNER_SET, globalNanoStep());

        if (validGhostZoneWidth == 0) {
            updateGhost();
            resetValidGhostZoneWidth();
        }
    }

    inline void notifyPatchAccepters(
        const Region<DIM>& region,
        const typename ParentType::PatchType& patchType,
        const long& nanoStep)
    {
        for (typename ParentType::PatchAccepterList::iterator i =
                 patchAccepters[patchType].begin();
             i != patchAccepters[patchType].end();
             ++i) {
            if (nanoStep == (*i)->nextRequiredNanoStep()) {
                (*i)->put(*oldGrid, region, nanoStep);
            }
        }
    }

    inline void notifyPatchProviders(
        const Region<DIM>& region,
        const typename ParentType::PatchType& patchType,
        const long& nanoStep)
    {
        for (typename ParentType::PatchProviderList::iterator i =
                 patchProviders[patchType].begin();
             i != patchProviders[patchType].end();
             ++i) {
            (*i)->get(
                &*oldGrid,
                region,
                nanoStep);
        }
    }

    inline long globalNanoStep() const
    {
        return curStep * CELL_TYPE::nanoSteps() + curNanoStep;
    }

    inline void initGrids()
    {
        Coord<DIM> topoDim = initializer->gridDimensions();
        CoordBox<DIM> gridBox;
        guessOffset(&gridBox.origin, &gridBox.dimensions);
        oldGrid.reset(new GridType(gridBox, CELL_TYPE(), CELL_TYPE(), topoDim));
        newGrid.reset(new GridType(gridBox, CELL_TYPE(), CELL_TYPE(), topoDim));
        initializer->grid(&*oldGrid);
        newGrid->getEdgeCell() = oldGrid->getEdgeCell();
        resetValidGhostZoneWidth();

        notifyPatchAccepters(
            rim(),
            ParentType::GHOST,
            globalNanoStep());
        notifyPatchAccepters(
            partitionManager->innerSet(0),
            ParentType::INNER_SET,
            globalNanoStep());

        kernelBuffer = PatchBufferType1(partitionManager->getVolatileKernel());
        rimBuffer = PatchBufferType2(rim());
        saveRim(globalNanoStep());
        updateGhost();
    }

    /**
     * computes the next ghost zone at time "t_1 = globalNanoStep() +
     * ghostZoneWidth()". Expects that oldGrid has its kernel and its
     * outer ghostzone updated to time "globalNanoStep()" and that the
     * inner ghostzones (rim) at time t_1 can be retrieved from the
     * internal patch buffer. Will leave oldgrid in a state so that
     * its whole ownRegion() will be at time t_1 and the rim will be
     * saved to the patchBuffer at "t2 = t1 + ghostZoneWidth()".
     */
    inline void updateGhost()
    {
        // fixme: skip all this ghost zone buffering for
        // ghostZoneWidth == 1?

        // 1: Prepare grid. The following update of the ghostzone will
        // destroy parts of the kernel, which is why we'll
        // save/restore those.
        saveKernel();
        // We need to restore the rim since it got destroyed while the
        // kernel was updated.
        restoreRim(false);

        // 2: actual ghostzone update
        int oldNanoStep = curNanoStep;
        int oldStep = curStep;
        int curGlobalNanoStep = globalNanoStep();

        for (int t = 0; t < ghostZoneWidth(); ++t) {
            notifyPatchProviders(
                partitionManager->rim(t), ParentType::GHOST, globalNanoStep());

            const Region<DIM>& region = partitionManager->rim(t + 1);
            for (typename Region<DIM>::StreakIterator i = region.beginStreak();
                 i != region.endStreak();
                 ++i) {
                UpdateFunctor<CELL_TYPE>()(
                    *i,
                    i->origin,
                    *oldGrid,
                    &*newGrid,
                    curNanoStep);
            }

            ++curNanoStep;
            if (curNanoStep == CELL_TYPE::nanoSteps()) {
                curNanoStep = 0;
                curStep++;
            }

            std::swap(oldGrid, newGrid);

            ++curGlobalNanoStep;
            notifyPatchAccepters(rim(), ParentType::GHOST, curGlobalNanoStep);
        }
        curNanoStep = oldNanoStep;
        curStep = oldStep;

        saveRim(curGlobalNanoStep);
        if (ghostZoneWidth() % 2) {
            std::swap(oldGrid, newGrid);
        }

        // 3: restore grid for kernel update
        restoreRim(true);
        restoreKernel();
    }

    inline const unsigned& ghostZoneWidth() const
    {
        return partitionManager->getGhostZoneWidth();
    }

    inline const Region<DIM>& rim() const
    {
        return partitionManager->rim(ghostZoneWidth());
    }

    inline void resetValidGhostZoneWidth()
    {
        validGhostZoneWidth = ghostZoneWidth();
    }

    inline void saveRim(const long& nanoStep)
    {
        rimBuffer.pushRequest(nanoStep);
        rimBuffer.put(*oldGrid, rim(), nanoStep);
    }

    inline void restoreRim(const bool& remove)
    {
        rimBuffer.get(&*oldGrid, rim(), globalNanoStep(), remove);
    }

    inline void saveKernel()
    {
        kernelBuffer.pushRequest(globalNanoStep());
        kernelBuffer.put(*oldGrid,
                         partitionManager->innerSet(ghostZoneWidth()),
                         globalNanoStep());
    }

    inline void restoreKernel()
    {
        kernelBuffer.get(
            &*oldGrid,
            partitionManager->getVolatileKernel(),
            globalNanoStep(),
            true);
    }
};

}
}

#endif
