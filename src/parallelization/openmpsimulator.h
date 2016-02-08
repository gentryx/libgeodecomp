#ifndef LIBGEODECOMP_PARALLELIZATION_OPENMPSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_OPENMPSIMULATOR_H

// include this file first to avoid clashes of Intel MPI with stdio.h.
#include <libgeodecomp/misc/apitraits.h>

#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/parallelization/monolithicsimulator.h>
#include <libgeodecomp/storage/gridtypeselector.h>
#include <libgeodecomp/storage/updatefunctor.h>

namespace LibGeoDecomp {

/**
 * OpenMPSimulator is based on SerialSimulator, but is capable of
 * threading via OpenMP.
 */
template<typename CELL_TYPE>
class OpenMPSimulator : public MonolithicSimulator<CELL_TYPE>
{
public:
    friend class OpenMPSimulatorTest;
    typedef typename MonolithicSimulator<CELL_TYPE>::GridType GridBaseType;
    typedef typename MonolithicSimulator<CELL_TYPE>::Topology Topology;
    typedef typename MonolithicSimulator<CELL_TYPE>::WriterVector WriterVector;
    typedef typename APITraits::SelectSoA<CELL_TYPE>::Value SupportsSoA;
    typedef typename GridTypeSelector<CELL_TYPE, Topology, false, SupportsSoA>::Value GridType;
    typedef typename Steerer<CELL_TYPE>::SteererFeedback SteererFeedback;

    static const int DIM = Topology::DIM;

    using MonolithicSimulator<CELL_TYPE>::NANO_STEPS;
    using MonolithicSimulator<CELL_TYPE>::chronometer;
    using MonolithicSimulator<CELL_TYPE>::initializer;
    using MonolithicSimulator<CELL_TYPE>::steerers;
    using MonolithicSimulator<CELL_TYPE>::stepNum;
    using MonolithicSimulator<CELL_TYPE>::writers;
    using MonolithicSimulator<CELL_TYPE>::getStep;
    using MonolithicSimulator<CELL_TYPE>::gridDim;

    /**
     * creates a OpenMPSimulator with the given initializer.
     */
    explicit OpenMPSimulator(
        Initializer<CELL_TYPE> *initializer,
        bool enableFineGrainedParallelism = false) :
        MonolithicSimulator<CELL_TYPE>(initializer),
        enableFineGrainedParallelism(enableFineGrainedParallelism)
    {
        stepNum = initializer->startStep();
        Coord<DIM> dim = initializer->gridBox().dimensions;
        curGrid = new GridType(CoordBox<DIM>(Coord<DIM>(), dim));
        newGrid = new GridType(CoordBox<DIM>(Coord<DIM>(), dim));
        initializer->grid(curGrid);
        initializer->grid(newGrid);

        // fixme: refactor OpenMPSimulator, serialsim, cudasim to reduce code duplication
        CoordBox<DIM> box = curGrid->boundingBox();
        simArea << box;
    }

    virtual ~OpenMPSimulator()
    {
        delete newGrid;
        delete curGrid;
    }

    /**
     * performs a single simulation step.
     */
    virtual void step()
    {
        SteererFeedback feedback;
        step(&feedback);
    }

    virtual void step(SteererFeedback *feedback)
    {
        TimeTotal t(&chronometer);

        handleInput(STEERER_NEXT_STEP, feedback);

        for (unsigned i = 0; i < NANO_STEPS; ++i) {
            nanoStep(i);
        }

        ++stepNum;
        handleOutput(WRITER_STEP_FINISHED);
    }

    /**
     * continue simulating until the maximum number of steps is reached.
     */
    virtual void run()
    {
        initializer->grid(curGrid);
        stepNum = initializer->startStep();
        setIORegions();

        SteererFeedback feedback;
        handleInput(STEERER_INITIALIZED, &feedback);
        handleOutput(WRITER_INITIALIZED);

        for (; stepNum < initializer->maxSteps();) {
            if (feedback.simulationEnded()) {
                break;
            }

            step(&feedback);
        }

        handleInput(STEERER_ALL_DONE, &feedback);
        handleOutput(WRITER_ALL_DONE);
    }

    /**
     * returns the current grid.
     */
    virtual const GridBaseType *getGrid()
    {
        return curGrid;
    }

protected:
    GridType *curGrid;
    GridType *newGrid;
    Region<DIM> simArea;
    bool enableFineGrainedParallelism;

    void nanoStep(unsigned nanoStep)
    {
        using std::swap;
        TimeCompute t(&chronometer);

        UpdateFunctor<CELL_TYPE, UpdateFunctorHelpers::ConcurrencyEnableOpenMP>()(
            simArea,
            Coord<DIM>(),
            Coord<DIM>(),
            *curGrid,
            newGrid,
            nanoStep,
            UpdateFunctorHelpers::ConcurrencyEnableOpenMP(true, enableFineGrainedParallelism));
        swap(curGrid, newGrid);
    }

    /**
     * notifies all registered Writers
     */
    void handleOutput(WriterEvent event)
    {
        TimeOutput t(&chronometer);

        for (unsigned i = 0; i < writers.size(); i++) {
            if ((event != WRITER_STEP_FINISHED) ||
                ((getStep() % writers[i]->getPeriod()) == 0)) {
                writers[i]->stepFinished(
                    *curGrid,
                    getStep(),
                    event);
            }
        }
    }

    /**
     * notifies all registered Steerers
     */
    void handleInput(SteererEvent event, SteererFeedback *feedback)
    {
        TimeInput t(&chronometer);

        for (unsigned i = 0; i < steerers.size(); ++i) {
            if ((event != STEERER_NEXT_STEP) ||
                (stepNum % steerers[i]->getPeriod() == 0)) {
                steerers[i]->nextStep(curGrid, simArea, gridDim, getStep(), event, 0, true, feedback);
            }
        }
        // fixme: apply SteererFeedback!
    }

    void setIORegions()
    {
        for (unsigned i = 0; i < steerers.size(); i++) {
            steerers[i]->setRegion(simArea);
        }
    }
};

}

#endif
