#ifndef LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_HPXSTEPPER_H
#define LIBGEODECOMP_PARALLELIZATION_HPXSIMULATOR_HPXSTEPPER_H

#include <libgeodecomp/parallelization/hiparsimulator/stepper.h>
#include <libgeodecomp/storage/patchbufferfixed.h>
#include <libgeodecomp/storage/updatefunctor.h>

#include <hpx/async.hpp>

namespace LibGeoDecomp {
namespace HiParSimulator {

template<typename CELL_TYPE>
class HpxStepper : public Stepper<CELL_TYPE>
{
    friend class HpxStepperRegionTest;
    friend class HpxStepperBasicTest;
    friend class HpxStepperTest;
public:
    typedef typename Stepper<CELL_TYPE>::Topology Topology;
    const static int DIM = Topology::DIM;
    const static unsigned NANO_STEPS = APITraits::SelectNanoSteps<CELL_TYPE>::VALUE;

    typedef class Stepper<CELL_TYPE> ParentType;
    typedef typename ParentType::GridType GridType;
    typedef PartitionManager<Topology> PartitionManagerType;
    typedef PatchBufferFixed<GridType, GridType, 1> PatchBufferType1;
    typedef PatchBufferFixed<GridType, GridType, 2> PatchBufferType2;
    typedef typename ParentType::PatchAccepterVec PatchAccepterVec;

    using Stepper<CELL_TYPE>::addPatchAccepter;
    using Stepper<CELL_TYPE>::chronometer;
    using Stepper<CELL_TYPE>::initializer;
    using Stepper<CELL_TYPE>::guessOffset;
    using Stepper<CELL_TYPE>::patchAccepters;
    using Stepper<CELL_TYPE>::patchProviders;
    using Stepper<CELL_TYPE>::partitionManager;

    inline HpxStepper(
        boost::shared_ptr<PartitionManagerType> partitionManager,
        boost::shared_ptr<Initializer<CELL_TYPE> > initializer,
        const PatchAccepterVec& ghostZonePatchAccepters = PatchAccepterVec(),
        const PatchAccepterVec& innerSetPatchAccepters = PatchAccepterVec()) :
        ParentType(partitionManager, initializer),
        asyncThreshold(boost::lexical_cast<std::size_t>(hpx::get_config_entry("LibGeoDecomp.asyncThreshold", "0")))
    {
        curStep = initializer->startStep();
        curNanoStep = 0;

        for (std::size_t i = 0; i < ghostZonePatchAccepters.size(); ++i) {
            addPatchAccepter(ghostZonePatchAccepters[i], ParentType::GHOST);
        }
        for (std::size_t i = 0; i < innerSetPatchAccepters.size(); ++i) {
            addPatchAccepter(innerSetPatchAccepters[i], ParentType::INNER_SET);
        }

        hpx::async(&HpxStepper::initGrids, this).wait();
    }

    inline void update(std::size_t nanoSteps)
    {
        for (std::size_t i = 0; i < nanoSteps; ++i)
        {
            update().wait();
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
    std::size_t asyncThreshold;
    std::size_t curStep;
    std::size_t curNanoStep;
    unsigned validGhostZoneWidth;
    boost::shared_ptr<GridType> oldGrid;
    boost::shared_ptr<GridType> newGrid;
    std::shared_ptr<PatchBufferType2> rimBuffer;
    std::shared_ptr<PatchBufferType1> kernelBuffer;
    Region<DIM> kernelFraction;
    hpx::lcos::local::spinlock gridMutex;
    double startTimeUpdate;

    inline hpx::future<void> update()
    {
        startTimeUpdate = ScopedTimer::time();

        unsigned index = ghostZoneWidth() - --validGhostZoneWidth;
        Region<DIM> region = partitionManager->innerSet(index);

        std::vector<hpx::future<void> > updateFutures;
        updateFutures.reserve(region.numStreaks());
        for (typename Region<DIM>::StreakIterator i = region.beginStreak();
             i != region.endStreak();
             ++i) {
            Region<DIM> r;
            r << *i;
            Coord<DIM> null;
            if(size_t(i->length()) > asyncThreshold) {
                updateFutures.push_back(
                    hpx::async(
                        UpdateFunctor<CELL_TYPE>(),
                        r,
                        null,
                        null,
                        boost::cref(*oldGrid),
                        &*newGrid,
                        curNanoStep
                    )
                );
            }
            else {
                UpdateFunctor<CELL_TYPE>()(r, null, null, *oldGrid, &*newGrid, curNanoStep);
            }
        }
        if(updateFutures.empty()) {
            updateFutures.push_back(hpx::make_ready_future());
        }

        ++curNanoStep;
        if (curNanoStep == NANO_STEPS) {
            curNanoStep = 0;
            curStep++;
        }

        return
            hpx::when_all(updateFutures).then(
                hpx::util::bind(&HpxStepper::updateGhostZones, this, region)
            );
    }

    // remove this function and compilation with g++ 4.7.3 fails. scary!
    template<typename T>
    void tock(double t)
    {
        std::cout << "toc(" << t << ")\n";
    }


    void updateGhostZones(const Region<DIM>& region)
    {
        std::swap(oldGrid, newGrid);
        chronometer.tock<TimeComputeInner>(startTimeUpdate);

        notifyPatchAccepters(region, ParentType::INNER_SET, globalNanoStep());

        if (validGhostZoneWidth == 0) {
            updateGhost();
            resetValidGhostZoneWidth();
        }

        notifyPatchProviders(region, ParentType::INNER_SET, globalNanoStep());
    }

    inline void notifyPatchAccepters(
        const Region<DIM>& region,
        const typename ParentType::PatchType& patchType,
        std::size_t nanoStep)
    {
        TimePatchAccepters t(&chronometer);

        std::vector<hpx::future<void> > patchAcceptersFutures;
        patchAcceptersFutures.reserve(patchAccepters[patchType].size());

        for (typename ParentType::PatchAccepterList::iterator i =
                 patchAccepters[patchType].begin();
             i != patchAccepters[patchType].end();
             ++i) {
            if (nanoStep == (*i)->nextRequiredNanoStep()) {
                patchAcceptersFutures.push_back(
                    hpx::async(
                        hpx::util::bind(
                            &PatchAccepter<GridType>::put,
                            *i,
                            boost::cref(*oldGrid),
                            region,
                            nanoStep
                        )
                    )
                );
            }
        }

        hpx::wait_all(patchAcceptersFutures);
    }

    inline void notifyPatchProviders(
        const Region<DIM>& region,
        const typename ParentType::PatchType& patchType,
        std::size_t nanoStep)
    {
        TimePatchAccepters t(&chronometer);

        std::vector<hpx::future<void> > patchProvidersFutures;
        patchProvidersFutures.reserve(patchProviders[patchType].size());
        void (PatchProvider<GridType>::*get)(
            GridType*,
            const Region<DIM>&,
            const std::size_t,
            hpx::lcos::local::spinlock&,
            bool) = &PatchProvider<GridType>::get;

        for (typename ParentType::PatchProviderList::iterator i =
                 patchProviders[patchType].begin();
             i != patchProviders[patchType].end();
             ++i) {
                patchProvidersFutures.push_back(
                    hpx::async(
                        hpx::util::bind(
                            get,
                            *i,
                            &*oldGrid,
                            region,
                            nanoStep,
                            boost::ref(gridMutex),
                            true
                        )
                    )
                );
        }

        hpx::wait_all(patchProvidersFutures);
    }

    inline std::size_t globalNanoStep() const
    {
        return curStep * NANO_STEPS + curNanoStep;
    }

    inline void initGrids()
    {
        const Coord<DIM>& topoDim = initializer->gridDimensions();
        CoordBox<DIM> gridBox;
        guessOffset(&gridBox.origin, &gridBox.dimensions);
        oldGrid.reset(new GridType(gridBox, CELL_TYPE(), CELL_TYPE(), topoDim));
        newGrid.reset(new GridType(gridBox, CELL_TYPE(), CELL_TYPE(), topoDim));
        initializer->grid(&*oldGrid);
        newGrid->getEdgeCell() = oldGrid->getEdgeCell();
        resetValidGhostZoneWidth();

        const Region<DIM>& rimRegion = rim();
        notifyPatchAccepters(
            rimRegion,
            ParentType::GHOST,
            globalNanoStep());
        notifyPatchAccepters(
            partitionManager->innerSet(0),
            ParentType::INNER_SET,
            globalNanoStep());

        kernelBuffer.reset(new PatchBufferType1(partitionManager->getVolatileKernel()));
        rimBuffer.reset(new PatchBufferType2(rimRegion));
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
        {
            TimeComputeGhost t(&chronometer);

            // fixme: skip all this ghost zone buffering for
            // ghostZoneWidth == 1?

            // 1: Prepare grid. The following update of the ghostzone will
            // destroy parts of the kernel, which is why we'll
            // save/restore those.
            saveKernel();
            // We need to restore the rim since it got destroyed while the
            // kernel was updated.
            restoreRim(false);
        }

        // 2: actual ghostzone update
        std::size_t oldNanoStep = curNanoStep;
        std::size_t oldStep = curStep;
        std::size_t curGlobalNanoStep = globalNanoStep();

        for (std::size_t t = 0; t < ghostZoneWidth(); ++t) {
            notifyPatchProviders(
                partitionManager->rim(t), ParentType::GHOST, globalNanoStep());

            {
                TimeComputeGhost timer(&chronometer);

                const Region<DIM>& region = partitionManager->rim(t + 1);

                std::vector<hpx::future<void> > updateFutures;
                updateFutures.reserve(region.numStreaks());
                for (typename Region<DIM>::StreakIterator i = region.beginStreak();
                     i != region.endStreak();
                     ++i) {
                    Region<DIM> r;
                    r << *i;
                    Coord<DIM> null;
                    if (size_t(i->length()) > asyncThreshold) {
                        updateFutures.push_back(
                            hpx::async(
                                UpdateFunctor<CELL_TYPE>(),
                                r,
                                null,
                                null,
                                boost::cref(*oldGrid),
                                &*newGrid,
                                curNanoStep
                                       )
                                                );
                    } else {
                        UpdateFunctor<CELL_TYPE>()(r, null, null, *oldGrid, &*newGrid, curNanoStep);
                    }
                }

                if (updateFutures.empty()) {
                    updateFutures.push_back(hpx::make_ready_future());
                }

                ++curNanoStep;
                if (curNanoStep == NANO_STEPS) {
                    curNanoStep = 0;
                    curStep++;
                }

                hpx::wait_all(updateFutures);
                std::swap(oldGrid, newGrid);

                ++curGlobalNanoStep;
            }

            notifyPatchAccepters(rim(), ParentType::GHOST, curGlobalNanoStep);
        }

        {
            TimeComputeGhost t(&chronometer);

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
    }
private:

    inline unsigned ghostZoneWidth() const
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

    inline void saveRim(std::size_t nanoStep)
    {
        rimBuffer->pushRequest(nanoStep);
        rimBuffer->put(*oldGrid, rim(), nanoStep);
    }

    inline void restoreRim(bool remove)
    {
        rimBuffer->get(&*oldGrid, rim(), globalNanoStep(), remove);
    }

    inline void saveKernel()
    {
        kernelBuffer->pushRequest(globalNanoStep());
        kernelBuffer->put(*oldGrid,
                         partitionManager->innerSet(ghostZoneWidth()),
                         globalNanoStep());
    }

    inline void restoreKernel()
    {
        kernelBuffer->get(
            &*oldGrid,
            partitionManager->getVolatileKernel(),
            globalNanoStep(),
            true);
    }
};

}
}

#endif
