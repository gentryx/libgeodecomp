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
#include <cuda.h>
#include <sstream>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class AutotuningSimulatorWithCudaTest : public CxxTest::TestSuite
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

    void testAddOwnSimulationsForCudaSim()
    {
        // fixme
        return;

#ifdef LIBGEODECOMP_WITH_THREADS
#ifdef LIBGEODECOMP_WITH_CPP14
        AutoTuningSimulator<SimFabTestCell, PatternOptimizer> ats(
            new SimFabTestInitializer(dim, maxSteps));
        ats.simulations.clear();
        SimulationParameters params;
        params.addParameter("BlockDimX", 1, 128);
        params.addParameter("BlockDimY", 1, 8);
        params.addParameter("BlockDimZ", 1, 8);
        ats.addNewSimulation("1.CudaSimulator",
            "CudaSimulation");
        ats.setParameters(params, "1.CudaSimulator");
        ats.run();
#endif
#endif
    }

    void testManuallyParamterizedCudaSim()
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

        ats.setParameters(params, "CudaSimulation");
        ats.run();
#endif
    }

    void testInvalidArgumentsForCudaSim()
    {
        // fixme
        return;

#ifdef LIBGEODECOMP_WITH_THREADS
#ifdef LIBGEODECOMP_WITH_CPP14
        AutoTuningSimulator<SimFabTestCell, PatternOptimizer> ats(
            new SimFabTestInitializer(dim, maxSteps));
        // This test don't test SimulationParameters!!!!
        SimulationParameters params;
        params.addParameter("BlockDimX", 1, 64);
        params.addParameter("BlockDimY", 1, 6);
        params.addParameter("BlockDimZ", 1, 6);
        // A JuliaSimulator is not valid
        TS_ASSERT_THROWS(ats.addNewSimulation("1.CudaSimulator",
            "JuliaSimulation"), std::invalid_argument);
        TS_ASSERT_THROWS(ats.setParameters(params, "1.CudaSimulator"),
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


// fixme: split into separate files
class SimulationFactoryWithCudaTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        dim = Coord<3>(100,100,100);
        maxSteps = 100;
        initializerProxy.reset(new VarStepInitializerProxy<SimFabTestCell>(
                                   new SimFabTestInitializer(dim,maxSteps)));
        cudaFab = new CudaSimulationFactory<SimFabTestCell>(initializerProxy);
#ifdef LIBGEODECOMP_WITH_THREADS
        cFab = new CacheBlockingSimulationFactory<SimFabTestCell>(initializerProxy);
#endif
        fab = new SerialSimulationFactory<SimFabTestCell>(initializerProxy);
#endif
    }

    void tearDown()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        delete cudaFab;
        delete fab;
#ifdef LIBGEODECOMP_WITH_THREADS
        delete cFab;
#endif
#endif
    }

    void testVarStepInitializerProxy()
    {
        // fixme
        return;

#ifdef LIBGEODECOMP_WITH_CPP14
        unsigned maxSteps = initializerProxy->maxSteps();
        double oldFitness = DBL_MIN;
        double aktFitness = 0.0;
        for (unsigned i = 10; i < maxSteps; i *= 2) {
            LOG(Logger::DBG, "setMaxSteps("<<i<<")")
            initializerProxy->setMaxSteps(i);
            LOG(Logger::DBG,"i: "<< i << " maxSteps(): "
                << initializerProxy->maxSteps() << " getMaxSteps(): "
                << initializerProxy->getMaxSteps())
            TS_ASSERT_EQUALS(i,initializerProxy->maxSteps());
            TS_ASSERT_EQUALS(i,initializerProxy->getMaxSteps());
            aktFitness = fab->operator()(fab->parameters());
            LOG(Logger::DBG, "Fitness: " << aktFitness)
            TS_ASSERT(oldFitness > aktFitness);
            oldFitness = aktFitness;
        }
        LOG(Logger::DBG, "getInitializer()->maxSteps(): "
                        << initializerProxy->getInitializer()->maxSteps()
                        << " \"initial\" maxSteps: " << maxSteps)
        TS_ASSERT_EQUALS(initializerProxy->getInitializer()->maxSteps(), maxSteps);
#endif
    }

    void testBasic()
    {
        // fixme
        return;

#ifdef LIBGEODECOMP_WITH_CPP14
        for (int i = 1; i <= 2; ++i) {
            Simulator<SimFabTestCell> *sim = fab->operator()();
            sim->run();
            delete sim;
        }
#endif
    }

    void testCacheBlockingFitness()
    {
        // fixme
        return;

#ifdef LIBGEODECOMP_WITH_THREADS
#ifdef LIBGEODECOMP_WITH_CPP14
        for (int i = 1; i <= 2; ++i) {
            cFab->parameters()["PipelineLength"].setValue(1);
            cFab->parameters()["WavefrontWidth"].setValue(100);
            cFab->parameters()["WavefrontHeight"].setValue(40);
            double fitness = cFab->operator()(cFab->parameters());
        }
#endif
#endif
    }

    void testCudaFitness()
    {
        // fixme
        return;

#ifdef LIBGEODECOMP_WITH_CPP14
        for (int i = 1; i <=2; ++i) {
            cudaFab->parameters()["BlockDimX"].setValue(15);
            cudaFab->parameters()["BlockDimY"].setValue(6);
            cudaFab->parameters()["BlockDimZ"].setValue(6);
            double fitness = cudaFab->operator()(cudaFab->parameters());
        }
#endif
    }

    void testAddWriterToSimulator()
    {
        // fixme
        return;

#ifdef LIBGEODECOMP_WITH_THREADS
#ifdef LIBGEODECOMP_WITH_CPP14
        MonolithicSimulator<SimFabTestCell> *sim = dynamic_cast<MonolithicSimulator<SimFabTestCell>*>((*cFab)());
        std::ostringstream buf;
        sim->addWriter(new TracingWriter<SimFabTestCell>(1, 100, 0, buf));
        sim->run();
        double fitness = cFab->operator()(cFab->parameters());
#endif
#endif
    }

    void testAddWriterToSerialSimulationFactory()
    {
        // fixme
        return;

#ifdef LIBGEODECOMP_WITH_CPP14
        std::ostringstream buf;
        Writer<SimFabTestCell> *writer = new TracingWriter<SimFabTestCell>(1, 100, 0, buf);
        fab->addWriter(*writer);
        fab->operator()(fab->parameters());
        delete writer;
#endif
    }

    void testAddWriterToCacheBlockingSimulationFactory()
    {
        // fixme
        return;

#ifdef LIBGEODECOMP_WITH_CPP14
#ifdef LIBGEODECOMP_WITH_THREADS
        std::ostringstream buf;
        Writer<SimFabTestCell> *writer = new TracingWriter<SimFabTestCell>(1, 100, 0, buf);
        cFab->addWriter(*writer);
        cFab->operator()(cFab->parameters());
        delete writer;
#endif
#endif
    }

    void testAddWriterToCudaSimulationFactory()
    {
        // fixme
        return;
#ifdef LIBGEODECOMP_WITH_CPP14
        std::ostringstream buf;
        Writer<SimFabTestCell> *writer = new TracingWriter<SimFabTestCell>(10, 100, 0, buf);
        cudaFab->addWriter(*writer);
        cudaFab->operator()(cudaFab->parameters());
        delete writer;
#endif
    }

private:

#ifdef LIBGEODECOMP_WITH_CPP14
    Coord<3> dim;
    unsigned maxSteps;
    boost::shared_ptr<VarStepInitializerProxy<SimFabTestCell> > initializerProxy;
    SimulationFactory<SimFabTestCell> *fab;
    SimulationFactory<SimFabTestCell> *cudaFab;

#ifdef LIBGEODECOMP_WITH_THREADS
    SimulationFactory<SimFabTestCell> *cFab;
#endif

#endif

};

}
