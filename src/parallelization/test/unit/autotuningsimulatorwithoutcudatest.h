#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/io/mpiiowriter.h>
#include <libgeodecomp/io/ppmwriter.h>
#include <libgeodecomp/io/simplecellplotter.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/misc/patternoptimizer.h>
#include <libgeodecomp/misc/simplexoptimizer.h>
#include <libgeodecomp/misc/simulationparameters.h>
#include <libgeodecomp/misc/simulationfactory.h>
#include <libgeodecomp/parallelization/autotuningsimulator.h>
#include <libgeodecomp/parallelization/cacheblockingsimulator.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <boost/assign/list_of.hpp>
#include <sstream>

using namespace LibGeoDecomp;

class SimFabTestCell
{
public:
    class API :
        public APITraits::HasFixedCoordsOnlyUpdate,
        public APITraits::HasStencil<Stencils::VonNeumann<3, 1> >,
        public APITraits::HasTorusTopology<3>,
        public APITraits::HasSeparateCUDAUpdate,
        public APITraits::HasPredefinedMPIDataType<double>
    {};

    inline explicit SimFabTestCell(double v = 0) : temp(v)
    {}

    template<typename COORD_MAP>
    __host__
    void update(const COORD_MAP& neighborhood, const unsigned& nanoStep)
    {
        temp = (neighborhood[FixedCoord< 0,  0, -1>()].temp +
                neighborhood[FixedCoord< 0, -1,  0>()].temp +
                neighborhood[FixedCoord<-1,  0,  0>()].temp +
                neighborhood[FixedCoord< 1,  0,  0>()].temp +
                neighborhood[FixedCoord< 0,  1,  0>()].temp +
                neighborhood[FixedCoord< 0,  0,  1>()].temp) * (1.0 / 6.0);
    }

    template<typename COORD_MAP>
    __device__
    void updateCuda(const COORD_MAP& neighborhood, const unsigned& nanoStep)
    {
        temp = (neighborhood[FixedCoord< 0,  0, -1>()].temp +
                neighborhood[FixedCoord< 0, -1,  0>()].temp +
                neighborhood[FixedCoord<-1,  0,  0>()].temp +
                neighborhood[FixedCoord< 1,  0,  0>()].temp +
                neighborhood[FixedCoord< 0,  1,  0>()].temp +
                neighborhood[FixedCoord< 0,  0,  1>()].temp) * (1.0 / 6.0);
    }

    double temp;
};

class SimFabTestInitializer : public SimpleInitializer<SimFabTestCell>
{
public:
    SimFabTestInitializer(Coord<3> gridDimensions = Coord<3>(100,100,100), unsigned timeSteps = 100) :
        SimpleInitializer<SimFabTestCell>(gridDimensions, timeSteps)
    {}

    virtual void grid(GridBase<SimFabTestCell, 3> *target)
    {
        int counter = 0;

        CoordBox<3> box = target->boundingBox();
        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            target->set(*i, SimFabTestCell(++counter));
        }
    }
};

class AutotuningSimulatorTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        dim = Coord<3>(100,100,100);
        maxSteps = 20;
    }

    void tearDown()
    {}

    void testBasicPatternOptimized()
    {
        LOG(Logger::INFO, "AutotuningSimulatorTest::TestBasicPatternOptimized()")
        AutoTuningSimulator<SimFabTestCell, PatternOptimizer> ats(
            new SimFabTestInitializer(dim, maxSteps));
        ats.setSimulationSteps(10);
        ats.run();
        std::vector<std::string> names = ats.getSimulationNames();
        for (std::vector<std::string>::iterator iter = names.begin();
            iter != names.end(); iter++)
            LOG(Logger::INFO, "Name: " << *iter << " Fitness: "
            << ats.getFitness(*iter)<< std::endl
            << ats.getSimulationParameters(*iter))
    }

    void testBasicSimplexOptimized()
    {
        LOG(Logger::INFO, "AutotuningSimulatorTest::testBasicSimplexOptimized()")
        AutoTuningSimulator<SimFabTestCell, SimplexOptimizer> ats(
            new SimFabTestInitializer(dim, maxSteps));
        ats.setSimulationSteps(10);
        ats.run();
        std::vector<std::string> names = ats.getSimulationNames();
        for (std::vector<std::string>::iterator iter = names.begin();
            iter != names.end(); iter++)
            LOG(Logger::INFO, "Name: " << *iter << " Fitness: "
            << ats.getFitness(*iter)<< std::endl
            << ats.getSimulationParameters(*iter))
    }

    void testAddOwnSimulations()
    {
#ifdef LIBGEODECOMP_WITH_THREADS
        LOG(Logger::INFO, "AutotuningSimulationTest::testAddOwnSimulations()")
        AutoTuningSimulator<SimFabTestCell, PatternOptimizer> ats(
            new SimFabTestInitializer(dim, maxSteps));
        ats.deleteAllSimulations();
        SimulationParameters params;
        params.addParameter("WavefrontWidth", 1, 300);
        params.addParameter("WavefrontHeight", 1, 300);
        params.addParameter("PipelineLength", 1, 25);
        ats.addNewSimulation("1.CacheBlockingSimulator",
            "CacheBlockingSimulation");
        ats.setParameters(params, "1.CacheBlockingSimulator");
        ats.run();
        std::vector<std::string> names = ats.getSimulationNames();

        for (std::vector<std::string>::iterator iter = names.begin(); iter != names.end(); iter++) {
            LOG(Logger::INFO, "Name: " << *iter << " Fitness: "
                << ats.getFitness(*iter)<< std::endl
                << ats.getSimulationParameters(*iter));
        }
#endif
    }

    void testManuallyParamterized()
    {
#ifdef LIBGEODECOMP_WITH_THREADS
        LOG(Logger::INFO, "AutotuningSimulatorTest:test:ManuallyParameterized()")
        AutoTuningSimulator<SimFabTestCell, PatternOptimizer> ats(
            new SimFabTestInitializer(dim, maxSteps));

        SimulationParameters params;
        params.addParameter("WavefrontWidth", 1, 300);
        params.addParameter("WavefrontHeight", 1, 300);
        params.addParameter("PipelineLength", 1, 25);

        ats.setParameters(params, "CacheBlockingSimulation");
        ats.run();

        std::vector<std::string> names = ats.getSimulationNames();
        for (std::vector<std::string>::iterator iter = names.begin(); iter != names.end(); iter++) {
            LOG(Logger::INFO, "Name: " << *iter << " Fitness: "
                << ats.getFitness(*iter)<< std::endl
                << ats.getSimulationParameters(*iter));
        }
#endif
    }

    void testInvalidArguments()
    {
#ifdef LIBGEODECOMP_WITH_THREADS
        LOG(Logger::INFO, "AutotuningSimulatorTest:testInvalidArguments()")
        AutoTuningSimulator<SimFabTestCell, PatternOptimizer> ats(
            new SimFabTestInitializer(dim, maxSteps));
        // This test don't test SimulationParameters!!!!
        SimulationParameters params;
        params.addParameter("WavefrontWidth", 1, 300);
        params.addParameter("WavefrontHeight", 1, 300);
        params.addParameter("PipelineLength", 1, 25);

        TS_ASSERT_THROWS(
            ats.addNewSimulation("1.CacheBlockingSimulator",
                                 "CachBlockingSimulation"),
            std::invalid_argument);
        TS_ASSERT_THROWS(
            ats.setParameters(params, "1.CacheBlockingSimulator"),
            std::invalid_argument);
        TS_ASSERT_THROWS(ats.getFitness("NoSimulator"), std::invalid_argument);
        TS_ASSERT_THROWS(ats.getSimulationParameters("NoSimulator"), std::invalid_argument);
        TS_ASSERT_THROWS(ats.setParameters(params, "NoSimulator"), std::invalid_argument);
#endif
    }

    void testAddWriter()
    {
        LOG(Logger::INFO, "AutotuningSimulatorTest::testAddWriter()")
        AutoTuningSimulator<SimFabTestCell, PatternOptimizer> ats(
            new SimFabTestInitializer(dim, maxSteps));
        ats.deleteAllSimulations();
        ats.addNewSimulation(
            "addWriterTest",
            "SerialSimulation");
        std::ostringstream buf;
        ats.addWriter(static_cast<Writer<SimFabTestCell> *>(new TracingWriter<SimFabTestCell>(1, 100, 0, buf)));
        ats.run();
    }
private:
    Coord<3> dim;
    unsigned maxSteps;
};

#include <libgeodecomp/io/clonableinitializer.h>
#include <libgeodecomp/io/varstepinitializerproxy.h>

class SimulationFactoryTest : public CxxTest::TestSuite
{

public:
    void setUp()
    {
        dim = Coord<3>(100,100,100);
        maxSteps = 100;
		initializerProxy = new VarStepInitializerProxy<SimFabTestCell>(new SimFabTestInitializer(dim,maxSteps));
#ifdef LIBGEODECOMP_WITH_THREADS
        cfab = new CacheBlockingSimulationFactory<SimFabTestCell>(
                    initializerProxy);
#endif
		fab = new SerialSimulationFactory<SimFabTestCell>(
                    initializerProxy);
    }

    void tearDown()
    {
        delete cfab;
        delete fab;
		delete initializerProxy;
    }

    void testBasic()
    {
        LOG(Logger::INFO, "SimulationFactoryTest::testBasic")
        for (int i =1;i<=2;i++)
        {
            Simulator<SimFabTestCell> *sim = fab->operator()();
            sim->run();
            delete sim;
        }
    }

    void testCacheBlockingFitness()
    {
#ifdef LIBGEODECOMP_WITH_THREADS
        LOG(Logger::INFO, "SimulationFactoryTest::testCacheBlockingFitness()")
        for (int i = 1; i <= 2; ++i) {
            cfab->parameters()["PipelineLength"].setValue(1);
            cfab->parameters()["WavefrontWidth"].setValue(100);
            cfab->parameters()["WavefrontHeight"].setValue(40);
            double fitness = cfab->operator()(cfab->parameters());
            LOG(Logger::INFO,  i << " fitness: " << fitness )
        }
#endif
    }

    void testAddWriterToSimulator()
    {
#ifdef LIBGEODECOMP_WITH_THREADS
        LOG(Logger::INFO, "SimulationFactoryTest::testAddWriterToSimulator()")
        CacheBlockingSimulator<SimFabTestCell> *sim =  (
            CacheBlockingSimulator<SimFabTestCell> *)cfab->operator()();
        std::ostringstream buf;
        sim->addWriter(new TracingWriter<SimFabTestCell>(1, 100, 0, buf));
        sim->run();
        double fitness = cfab->operator()(cfab->parameters());
        LOG(Logger::INFO, "Fitness: " << fitness << std::endl)
#endif
    }

    void testAddWriterToSerialSimulationFactory()
    {
        LOG(Logger::INFO, "SimulationFactoryTest::testAddWriterToSerialSimulationFactory()")
        std::ostringstream buf;
        Writer<SimFabTestCell> *writer = new TracingWriter<SimFabTestCell>(1, 100, 0, buf);
        fab->addWriter(*writer);
        fab->operator()(fab->parameters());
        delete writer;
    }

    void testAddWriterToCacheBlockingSimulationFactory()
    {
#ifdef LIBGEODECOMP_WITH_THREADS
        LOG(Logger::INFO, "SimulationFactoryTest::testAddWriterToCacheBlockingSimulationFactory()")
        std::ostringstream buf;
        Writer<SimFabTestCell> *writer = new TracingWriter<SimFabTestCell>(1, 100, 0, buf);
        cfab->addWriter(*writer);
        cfab->operator()(cfab->parameters());
        delete writer;
#endif
    }

private:
    Coord<3> dim;
    unsigned maxSteps;
    VarStepInitializerProxy<SimFabTestCell> *initializerProxy;
	ClonableInitializer<SimFabTestCell> *initializerClonable;
	SimulationFactory<SimFabTestCell> *fab;
    SimulationFactory<SimFabTestCell> *cfab;
};
