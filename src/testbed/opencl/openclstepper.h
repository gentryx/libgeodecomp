#ifndef LIBGEODECOMP_TESTBED_OPENCL_OPENCLSTEPPER_H
#define LIBGEODECOMP_TESTBED_OPENCL_OPENCLSTEPPER_H

#include <libgeodecomp/storage/patchbufferfixed.h>
#include <libgeodecomp/storage/updatefunctor.h>
#include <libgeodecomp/parallelization/hiparsimulator/stepper.h>

#include <libgeodecomp/testbed/opencl/openclwrapper.h>

namespace LibGeoDecomp {

/**
 * Dead prototype
 */
template<typename CELL_TYPE, typename DATA_TYPE>
class OpenCLCellInterface {
public:
    static std::string kernel_file(void)
    {
        return CELL_TYPE::kernel_file();
    }

    static std::string kernel_function(void)
    {
        return CELL_TYPE::kernel_function();
    }

    virtual DATA_TYPE * data(void) = 0;
};

namespace HiParSimulator {

/**
 * Dead prototype
 */
template<typename CELL_TYPE, typename DATA_TYPE>
class OpenCLStepper : public Stepper<CELL_TYPE>
{
public:
    friend class OpenCLStepperRegionTest;
    friend class OpenCLStepperBasicTest;
    friend class OpenCLStepperTest;

    using Stepper<CELL_TYPE>::addPatchAccepter;
    using Stepper<CELL_TYPE>::initializer;
    using Stepper<CELL_TYPE>::guessOffset;
    using Stepper<CELL_TYPE>::patchAccepters;
    using Stepper<CELL_TYPE>::patchProviders;
    using Stepper<CELL_TYPE>::partitionManager;

    typedef typename APITraits::SelectTopology<CELL_TYPE>::Value Topology;
    typedef class Stepper<CELL_TYPE> ParentType;
    typedef typename ParentType::GridType GridType;
    typedef PartitionManager<Topology> PartitionManagerType;
    typedef PatchBufferFixed<GridType, GridType, 1> PatchBufferType1;
    typedef PatchBufferFixed<GridType, GridType, 2> PatchBufferType2;
    typedef typename ParentType::PatchAccepterVec PatchAccepterVec;

    const static int DIM = Topology::DIM;
    const static int NANO_STEPS = APITraits::SelectNanoSteps<CELL_TYPE>::VALUE;

    inline OpenCLStepper(
        unsigned platformID,
        unsigned deviceID,
        boost::shared_ptr<PartitionManagerType> partitionManager,
        boost::shared_ptr<Initializer<CELL_TYPE> > initializer,
        const PatchAccepterVec& ghostZonePatchAccepters = PatchAccepterVec(),
        const PatchAccepterVec& innerSetPatchAccepters = PatchAccepterVec()) :
        ParentType(partitionManager, initializer),
        platformID(platformID), deviceID(deviceID)
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

        std::string kernel_file = OpenCLCellInterface<CELL_TYPE, DATA_TYPE>::kernel_file();
        std::string kernel_function = OpenCLCellInterface<CELL_TYPE, DATA_TYPE>::kernel_function();

        oclwrapper = OpenCLWrapper_Ptr(
            new OpenCLWrapper<DATA_TYPE>(
                platformID, deviceID,
                kernel_file, kernel_function,
                initializer->gridBox().dimensions.x(),
                initializer->gridBox().dimensions.y(),
                initializer->gridBox().dimensions.z()));
    }

    inline virtual void update(std::size_t nanoSteps)
    {
        for (int i = 0; i < nanoSteps; ++i) {
            update();
        }
    }

    inline virtual std::pair<std::size_t, std::size_t> currentStep() const
    {
        return std::make_pair(curStep, curNanoStep);
    }

    inline virtual const GridType& grid() const
    {
        return *oldGrid;
    }

private:
    typedef std::shared_ptr<OpenCLWrapper<DATA_TYPE>> OpenCLWrapper_Ptr;

    unsigned int platformID;
    unsigned int deviceID;

    int curStep;
    int curNanoStep;
    int validGhostZoneWidth;
    boost::shared_ptr<GridType> oldGrid;
    boost::shared_ptr<GridType> newGrid;
    PatchBufferType2 rimBuffer;
    PatchBufferType1 kernelBuffer;
    Region<DIM> kernelFraction;

    OpenCLWrapper_Ptr oclwrapper;

    void copyGridToHost(void)
    {
        auto box = initializer->gridBox();
        int x_size = box.dimensions.x();
        int y_size = box.dimensions.y();

        DATA_TYPE * data = static_cast<DATA_TYPE *>(oclwrapper->readDeviceData());
        oclwrapper->finish();

        for (auto & p : box) {
            auto & cell = dynamic_cast<OpenCLCellInterface<CELL_TYPE, DATA_TYPE> &>((*newGrid)[p]);
            // using newGrid here, compared to data_to_device  ^^^

            uint32_t address = p.z() * y_size * x_size
                + p.y() * x_size
                + p.x();

            *(cell.data()) = data[address];
        }
    }

    void copyGridToDevice(void)
    {
        try {
            auto box = initializer->gridBox();

            std::vector<typename OpenCLWrapper<DATA_TYPE>::data_t> data;
            std::vector<typename OpenCLWrapper<DATA_TYPE>::point_t> points;

            for (auto & p : box) {
                auto & cell = dynamic_cast<OpenCLCellInterface<CELL_TYPE, DATA_TYPE> &>((*oldGrid)[p]);
                points.push_back(std::make_tuple(p.x(), p.y(), p.z()));
                data.push_back(cell.data());
            }

            oclwrapper->loadPoints(points.begin(), points.end());
            oclwrapper->loadHostData(data.begin(), data.end());
        } catch(std::exception & error) {
            std::cerr << __PRETTY_FUNCTION__ << ": " << error.what() << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    inline void update()
    {
        unsigned index = ghostZoneWidth() - --validGhostZoneWidth;
        const Region<DIM>& region = partitionManager->innerSet(index);

        copyGridToDevice();
        oclwrapper->run();
        oclwrapper->finish();
        copyGridToHost();
        std::swap(newGrid, oldGrid);

        ++curNanoStep;
        if (curNanoStep == NANO_STEPS) {
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
        return curStep * NANO_STEPS + curNanoStep;
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
            UpdateFunctor<CELL_TYPE>()(
                region,
                Coord<DIM>(),
                Coord<DIM>(),
                *oldGrid,
                &*newGrid,
                curNanoStep);

            ++curNanoStep;
            if (curNanoStep == NANO_STEPS) {
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

    inline const unsigned ghostZoneWidth() const
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
