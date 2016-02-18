#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/io/varstepinitializerproxy.h>
#include <libgeodecomp/misc/patternoptimizer.h>
#include <libgeodecomp/misc/simfabtestmodel.h>
#include <libgeodecomp/misc/simplexoptimizer.h>
#include <libgeodecomp/misc/simulationfactory.h>
#include <libgeodecomp/misc/simulationparameters.h>
#include <libgeodecomp/parallelization/autotuningsimulator.h>
#include <boost/assign/list_of.hpp>
#include <sstream>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class AutotuningSimulatorWithoutCudaTest : public CxxTest::TestSuite
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

    void testAddOwnSimulationsForCacheBlockingSim()
    {
        // fixme
        return;

#ifdef LIBGEODECOMP_WITH_THREADS
#ifdef LIBGEODECOMP_WITH_CPP14
        AutoTuningSimulator<SimFabTestCell, PatternOptimizer> ats(
            new SimFabTestInitializer(dim, maxSteps));
        SimulationParameters params;
        params.addParameter("WavefrontWidth", 1, 300);
        params.addParameter("WavefrontHeight", 1, 300);
        params.addParameter("PipelineLength", 1, 25);
        ats.addNewSimulation("1.CacheBlockingSimulator",
            "CacheBlockingSimulation");
        ats.setParameters(params, "1.CacheBlockingSimulator");
        ats.run();
#endif
#endif
    }

    void testManuallyParamterizedCacheBlockingSim()
    {
        // fixme
        return;

#ifdef LIBGEODECOMP_WITH_THREADS
#ifdef LIBGEODECOMP_WITH_CPP14
        AutoTuningSimulator<SimFabTestCell, PatternOptimizer> ats(
            new SimFabTestInitializer(dim, maxSteps));

        SimulationParameters params;
        params.addParameter("WavefrontWidth", 1, 300);
        params.addParameter("WavefrontHeight", 1, 300);
        params.addParameter("PipelineLength", 1, 25);

        ats.setParameters(params, "CacheBlockingSimulation");
        ats.run();
#endif
#endif
    }

    void testInvalidArgumentsForCacheBlockingSim()
    {
        // fixme
        return;

#ifdef LIBGEODECOMP_WITH_THREADS
#ifdef LIBGEODECOMP_WITH_CPP14
        AutoTuningSimulator<SimFabTestCell, PatternOptimizer> ats(
            new SimFabTestInitializer(dim, maxSteps));
        // This test don't test SimulationParameters!!!!
        SimulationParameters params;
        params.addParameter("WavefrontWidth", 1, 300);
        params.addParameter("WavefrontHeight", 1, 300);
        params.addParameter("PipelineLength", 1, 25);
        // The e in CacheBlockingSimlator is missing...
        TS_ASSERT_THROWS(
            ats.addNewSimulation("1.CacheBlockingSimulator",
                                 "CachBlockingSimulation"),
            std::invalid_argument);
        TS_ASSERT_THROWS(
            ats.setParameters(params, "1.CacheBlockingSimulator"),
            std::invalid_argument);
        TS_ASSERT_THROWS(ats.setParameters(params, "NoSimulator"), std::invalid_argument);
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
        ats.addNewSimulation(
            "SerialSimulation",
            "SerialSimulation");
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
