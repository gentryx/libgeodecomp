#ifndef LIBGEODECOMP_PARALLELIZATION_SERIALSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_SERIALSIMULATOR_H

#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/misc/updatefunctor.h>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/parallelization/monolithicsimulator.h>

namespace LibGeoDecomp {

/**
 * SerialSimulator is the simplest implementation of the Simulator
 * concept (or rather the MonolithicSimulator, to be exact). It's
 * purpose is to make fostering new applications easier. The absence
 * of concurrency simplifies debugging. As its name implies, it
 * doesn't do any threading, but vectorization (SIMD) is supported.
 */
template<typename CELL_TYPE>
class SerialSimulator : public MonolithicSimulator<CELL_TYPE>
{
public:
    friend class SerialSimulatorTest;
    typedef APITraits::SelectSoA<CELL_TYPE> GridTypeSelector;
    typedef typename MonolithicSimulator<CELL_TYPE>::GridType GridBaseType;
    typedef typename MonolithicSimulator<CELL_TYPE>::Topology Topology;
    typedef typename MonolithicSimulator<CELL_TYPE>::WriterVector WriterVector;
    typedef typename APITraits::SelectSoA<CELL_TYPE>::Value SupportsSoA;
    typedef typename SimulatorHelpers::GridTypeSelector<CELL_TYPE, Topology, false, SupportsSoA>::Value GridType;
    static const int DIM = Topology::DIM;

    using MonolithicSimulator<CELL_TYPE>::NANO_STEPS;
    using MonolithicSimulator<CELL_TYPE>::initializer;
    using MonolithicSimulator<CELL_TYPE>::steerers;
    using MonolithicSimulator<CELL_TYPE>::stepNum;
    using MonolithicSimulator<CELL_TYPE>::writers;
    using MonolithicSimulator<CELL_TYPE>::getStep;
    using MonolithicSimulator<CELL_TYPE>::gridDim;

    /**
     * creates a SerialSimulator with the given initializer.
     */
    SerialSimulator(Initializer<CELL_TYPE> *initializer) :
        MonolithicSimulator<CELL_TYPE>(initializer)
    {
        stepNum = initializer->startStep();
        Coord<DIM> dim = initializer->gridBox().dimensions;
        curGrid = new GridType(CoordBox<DIM>(Coord<DIM>(), dim));
        newGrid = new GridType(CoordBox<DIM>(Coord<DIM>(), dim));
        initializer->grid(curGrid);
        initializer->grid(newGrid);

        // fixme: refactor serialsim, cudasim to reduce code duplication
        CoordBox<DIM> box = curGrid->boundingBox();
        unsigned endX = box.dimensions.x();
        box.dimensions.x() = 1;
        for (typename CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
            simArea << Streak<DIM>(*i, endX);
        }
    }

    ~SerialSimulator()
    {
        delete newGrid;
        delete curGrid;
    }

    /**
     * performs a single simulation step.
     */
    virtual void step()
    {
        handleInput(STEERER_NEXT_STEP);

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

        handleInput(STEERER_INITIALIZED);
        handleOutput(WRITER_INITIALIZED);

        for (; stepNum < initializer->maxSteps();) {
            step();
        }

        handleInput(STEERER_ALL_DONE);
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

    void nanoStep(const unsigned& nanoStep)
    {
        UpdateFunctor<CELL_TYPE>()(simArea, Coord<DIM>(), *curGrid, newGrid, nanoStep);
        std::swap(curGrid, newGrid);
    }

    /**
     * notifies all registered Writers
     */
    void handleOutput(WriterEvent event)
    {
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
    void handleInput(SteererEvent event)
    {
        for (unsigned i = 0; i < steerers.size(); ++i) {
            if (stepNum % steerers[i]->getPeriod() == 0) {
                steerers[i]->nextStep(curGrid, simArea, gridDim, getStep(), event, 0, true);
            }
        }
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
