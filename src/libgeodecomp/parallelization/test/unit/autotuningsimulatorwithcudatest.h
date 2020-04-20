#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/io/varstepinitializerproxy.h>
#include <libgeodecomp/misc/patternoptimizer.h>
#include <libgeodecomp/misc/simfabtestmodel.h>
#include <libgeodecomp/misc/simplexoptimizer.h>
#include <libgeodecomp/misc/simulationparameters.h>
#include <libgeodecomp/parallelization/autotuningsimulator.h>
#include <cuda.h>
#include <sstream>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

/**
 * The tests in this class are virtially identical to the ones running
 * without CUDA. This duplication is acceptable as the same code may
 * behave differently in the presence of CUDA.
 */
class AutotuningSimulatorWithCUDATest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        dim = Coord<3>(25, 25, 25);
        maxSteps = 20;
    }

    void tearDown()
    {}

    void testBasicPatternOptimized()
    {
        // fixme
        return;

#ifdef LIBGEODECOMP_WITH_CPP14
        AutoTuningSimulator<SimFabTestCell, PatternOptimizer> ats(
            new SimFabTestInitializer(dim, maxSteps), 10);
        ats.run();
#endif
    }

    void testNormalizeSteps()
    {
        // fixme
        return;

#ifdef LIBGEODECOMP_WITH_CPP14
        const double goal = -0.4;
        AutoTuningSimulator<SimFabTestCell, PatternOptimizer> ats(
            new SimFabTestInitializer(dim, maxSteps));

        unsigned steps = ats.normalizeSteps(goal, 5);
        unsigned originalSteps = steps;
        LOG(Logger::DBG, "Result of nomalizeSteps(" << goal << ") is: " << steps)

        AutoTuningSimulator<SimFabTestCell, PatternOptimizer> ats2(
            new SimFabTestInitializer(dim, maxSteps));

        unsigned startValue = steps / 2;
        if (startValue < 2) {
            startValue = 2;
        }
        steps = ats2.normalizeSteps(goal, startValue);
        LOG(Logger::DBG, "Result of nomalizeSteps(" << goal << " ,"
            << startValue << ") is: " << steps);
        // TODO: Maybe the following assertions are too strong or
        // depend too much on the system where the test is running
        TS_ASSERT( steps <= originalSteps + 5 && steps >= originalSteps - 5 );
        startValue = steps + (steps / 2);
        steps = ats2.normalizeSteps(goal, startValue);
        LOG(Logger::DBG, "Result of nomalizeSteps(" << goal << " ,"
                    << startValue << ") is: " << steps)
        TS_ASSERT( steps <= originalSteps + 5 && steps >= originalSteps - 5);
#endif
    }

    void testBasicSimplexOptimized()
    {
        // fixme
        return;

#ifdef LIBGEODECOMP_WITH_CPP14
        AutoTuningSimulator<SimFabTestCell, SimplexOptimizer> ats(
            new SimFabTestInitializer(dim, maxSteps), 10);
        ats.run();
#endif
    }

    void testAddOwnSimulationsForCUDASim()
    {
        // fixme
        return;

#ifdef LIBGEODECOMP_WITH_THREADS
#ifdef LIBGEODECOMP_WITH_CPP14
        AutoTuningSimulator<SimFabTestCell, PatternOptimizer> ats(
            new SimFabTestInitializer(dim, maxSteps));
        ats.simulations.clear();
        ats.addSimulation("1.CUDASimulator", CUDASimulationFactory<SimFabTestCell>(ats.varStepInitializer));
        ats.run();
#endif
#endif
    }

    void testManuallyParamterizedCUDASim()
    {
        // fixme
        return;

#ifdef LIBGEODECOMP_WITH_CPP14
        AutoTuningSimulator<SimFabTestCell, PatternOptimizer> ats(
            new SimFabTestInitializer(dim, maxSteps));

        SimulationParameters params;
        params.addParameter("BlockDimX", 1, 64);
        params.addParameter("BlockDimY", 1, 6);
        params.addParameter("BlockDimZ", 1, 6);

        ats.getSimulation("CUDASimulator")->parameters = params;
        ats.run();
#endif
    }

    void testInvalidArgumentsForCUDASim()
    {
        // fixme
        return;

#ifdef LIBGEODECOMP_WITH_THREADS
#ifdef LIBGEODECOMP_WITH_CPP14
        AutoTuningSimulator<SimFabTestCell, PatternOptimizer> ats(
            new SimFabTestInitializer(dim, maxSteps));
        TS_ASSERT_THROWS(ats.getSimulation("1.CUDASimulator"), std::invalid_argument&);
        TS_ASSERT_THROWS(ats.getSimulation("NoSimulator"), std::invalid_argument&);
#endif
#endif
    }

    void testAddWriter()
    {
        // fixme
        return;

#ifdef LIBGEODECOMP_WITH_CPP14
        AutoTuningSimulator<SimFabTestCell, PatternOptimizer> ats(
            new SimFabTestInitializer(dim, maxSteps));
        ats.simulations.clear();
        ats.addSimulation(
            "SerialSimulator",
            SerialSimulationFactory<SimFabTestCell>(ats.varStepInitializer));

        std::ostringstream buf;
        ats.addWriter(static_cast<Writer<SimFabTestCell> *>(new TracingWriter<SimFabTestCell>(1, 100, 0, buf)));
        ats.run();
#endif
    }

private:
    Coord<3> dim;
    unsigned maxSteps;
};

}
