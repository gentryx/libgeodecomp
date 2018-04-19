#ifndef LIBGEODECOMP_PARALLELIZATION_OPENMPSIMULATOR_H
#define LIBGEODECOMP_PARALLELIZATION_OPENMPSIMULATOR_H

// include this file first to avoid clashes of Intel MPI with stdio.h.
#include <libgeodecomp/misc/apitraits.h>

#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/storage/gridtypeselector.h>
#include <libgeodecomp/storage/updatefunctor.h>

namespace LibGeoDecomp {

// padding is fine, as is (not) inlining.
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4710 4711 4820 )
#endif

/**
 * OpenMPSimulator is based on SerialSimulator, but is capable of
 * threading via OpenMP.
 */
template<typename CELL_TYPE>
class OpenMPSimulator : public SerialSimulator<CELL_TYPE>
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

    using SerialSimulator<CELL_TYPE>::NANO_STEPS;
    using SerialSimulator<CELL_TYPE>::chronometer;
    using SerialSimulator<CELL_TYPE>::curGrid;
    using SerialSimulator<CELL_TYPE>::initializer;
    using SerialSimulator<CELL_TYPE>::newGrid;
    using SerialSimulator<CELL_TYPE>::simArea;
    using SerialSimulator<CELL_TYPE>::steerers;
    using SerialSimulator<CELL_TYPE>::stepNum;
    using SerialSimulator<CELL_TYPE>::writers;
    using SerialSimulator<CELL_TYPE>::getStep;
    using SerialSimulator<CELL_TYPE>::gridDim;

    /**
     * creates a OpenMPSimulator with the given initializer.
     */
    explicit OpenMPSimulator(
        Initializer<CELL_TYPE> *initializer,
        bool enableFineGrainedParallelism = false) :
        SerialSimulator<CELL_TYPE>(initializer),
        enableFineGrainedParallelism(enableFineGrainedParallelism)
    {}

protected:
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

#pragma omp parallel for schedule(dynamic)
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

#pragma omp parallel for schedule(dynamic)
        for (unsigned i = 0; i < steerers.size(); ++i) {
            if ((event != STEERER_NEXT_STEP) ||
                (stepNum % steerers[i]->getPeriod() == 0)) {
                steerers[i]->nextStep(curGrid, simArea, gridDim, getStep(), event, 0, true, feedback);
            }
        }
    }
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif
