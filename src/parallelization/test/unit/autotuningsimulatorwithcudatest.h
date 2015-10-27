#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/misc/simulationfactory.h>
#include <libgeodecomp/misc/patternoptimizer.h>
#include <libgeodecomp/misc/simplexoptimizer.h>
#include <libgeodecomp/misc/simulationparameters.h>
#include <libgeodecomp/parallelization/autotuningsimulator.h>
#include <cuda.h>

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
namespace LibGeoDecomp {

class AutotuningSimulatorWithCudaTest : public CxxTest::TestSuite
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
        LOG(Logger::INFO, "AutotuningSimulatorWithCudaTest::TestBasicPatternOptimized()")
        AutoTuningSimulator<SimFabTestCell, PatternOptimizer> ats(
            SimFabTestInitializer(dim, maxSteps));
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
            SimFabTestInitializer(dim, maxSteps));
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
        LOG(Logger::INFO, "AutotuningSimulationTest::testAddOwnSimulations()")
        AutoTuningSimulator<SimFabTestCell, PatternOptimizer> ats(
            SimFabTestInitializer(dim, maxSteps));
        ats.deleteAllSimulations();
        SimulationParameters params;
        params.addParameter("WavefrontWidth", 1, 300);
        params.addParameter("WavefrontHeight", 1, 300);
        params.addParameter("PipelineLength", 1, 25);
        ats.addNewSimulation("1.CacheBlockingSimulator",
            "CacheBlockingSimulation",
            SimFabTestInitializer(dim,maxSteps));
        ats.setParameters(params, "1.CacheBlockingSimulator");
        ats.run();
        std::vector<std::string> names = ats.getSimulationNames();
        for (std::vector<std::string>::iterator iter = names.begin();
            iter != names.end(); iter++)
            LOG(Logger::INFO, "Name: " << *iter << " Fitness: " 
            << ats.getFitness(*iter)<< std::endl 
            << ats.getSimulationParameters(*iter))
    }

    void testManuallyParamterized()
    {
        LOG(Logger::INFO, "AutotuningSimulatorTest:test:ManuallyParameterized()")
        AutoTuningSimulator<SimFabTestCell, PatternOptimizer> ats(
            SimFabTestInitializer(dim, maxSteps));
        
        SimulationParameters params;
        params.addParameter("WavefrontWidth", 1, 300);
        params.addParameter("WavefrontHeight", 1, 300);
        params.addParameter("PipelineLength", 1, 25);
        
        ats.setParameters(params, "CacheBlockingSimulation");
        ats.run();
        
        std::vector<std::string> names = ats.getSimulationNames();
        for (std::vector<std::string>::iterator iter = names.begin();
            iter != names.end(); iter++)
            LOG(Logger::INFO, "Name: " << *iter << " Fitness: " 
            << ats.getFitness(*iter)<< std::endl 
            << ats.getSimulationParameters(*iter))
    }

    void testInvalidArguments()
    {
        LOG(Logger::INFO, "AutotuningSimulatorTest:testInvalidArguments()")
        AutoTuningSimulator<SimFabTestCell, PatternOptimizer> ats(
            SimFabTestInitializer(dim, maxSteps));
        // This test don't test SimulationParameters!!!!
        SimulationParameters params;
        params.addParameter("WavefrontWidth", 1, 300);
        params.addParameter("WavefrontHeight", 1, 300);
        params.addParameter("PipelineLength", 1, 25);
        TS_ASSERT_THROWS(ats.addNewSimulation("1.CacheBlockingSimulator",
            "CachBlockingSimulation",
            SimFabTestInitializer(dim,maxSteps)), std::invalid_argument);
        TS_ASSERT_THROWS(ats.setParameters(params, "1.CacheBlockingSimulator"),
            std::invalid_argument);
        TS_ASSERT_THROWS(ats.getFitness("NoSimulator"), std::invalid_argument);
        TS_ASSERT_THROWS(ats.getSimulationParameters("NoSimulator"), std::invalid_argument);
        TS_ASSERT_THROWS(ats.setParameters(params, "NoSimulator"), std::invalid_argument);

    }
    
    void testAddWriter()
    {
        LOG(Logger::INFO, "AutotuningSimulatorTest::testAddWriter()")
        AutoTuningSimulator<SimFabTestCell, PatternOptimizer> ats(
            SimFabTestInitializer(dim, maxSteps));
        ats.deleteAllSimulations();
        ats.addNewSimulation(
            "addWriterTest", 
            "SerialSimulation",
            SimFabTestInitializer(dim, maxSteps));
        ats.addWriter((Writer<SimFabTestCell> *) new TracingWriter<SimFabTestCell>(1,100));
        ats.run();
    }

private:
    Coord<3> dim;
    unsigned maxSteps;

};




class SimulationFactoryWithCudaTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        dim = Coord<3>(100,100,100);
        maxSteps = 100;
        cudaFab = new CudaSimulationFactory<SimFabTestCell>(SimFabTestInitializer(dim, maxSteps));
        cFab = new CacheBlockingSimulationFactory<SimFabTestCell>(
                    SimFabTestInitializer(dim, maxSteps));
        fab = new SerialSimulationFactory<SimFabTestCell>(
                    SimFabTestInitializer(dim, maxSteps));
    }
    
    void tearDown()
    {
        delete cudaFab;
        delete fab;
        delete cFab;
    }

    void testBasic()
    {
        LOG(Logger::INFO, "SimulationFactoryWithCudaTest::testBasic")
        for (int i =1; i <= 2; i++)
        {
            Simulator<SimFabTestCell> *sim = fab->operator()();
            sim->run();
            delete sim;
        }
    }

    void testCacheBlockingFitness()
    {
        LOG(Logger::INFO, "SimulationFactoryWithCudaTest::testCacheBlockingFitness()")
        for (int i = 1; i <= 2; i++){
            cFab->parameters()["PipelineLength"].setValue(1);
            cFab->parameters()["WavefrontWidth"].setValue(100);
            cFab->parameters()["WavefrontHeight"].setValue(40);
            double fitness = cFab->operator()(cFab->parameters());
            LOG(Logger::INFO,  i << " fitness: " << fitness )
        }
    }

    void testCudaFitness()
    {
        LOG(Logger::INFO, "SimulationFactoryWithCudaTest::testCudaFitness()")
        for (int i = 1; i <=2; ++i){
            cudaFab->parameters()["BlockDimX"].setValue(15);
            cudaFab->parameters()["BlockDimY"].setValue(6);
            cudaFab->parameters()["BlockDimZ"].setValue(6);
            double fitness = cudaFab->operator()(cudaFab->parameters());
            LOG(Logger::INFO, i << " fitness: " << fitness)
        }
    }

    void testAddWriterToSimulator()
    {
        LOG(Logger::INFO, "SimulationFactoryWithCudaTest::testAddWriterToSimulator()")
        CacheBlockingSimulator<SimFabTestCell> *sim =  (
            CacheBlockingSimulator<SimFabTestCell> *)cFab->operator()();
        sim->addWriter(new TracingWriter<SimFabTestCell>(1, 100));
        sim->run();
        double fitness = cFab->operator()(cFab->parameters());
        LOG(Logger::INFO, "Fitness: " << fitness << std::endl)
    }
 
    void testAddWriterToSerialSimulationFactory()
    {
        LOG(Logger::INFO, "SimulationFactoryWithCudaTest::testAddWriterToSerialSimulationFactory()")
        Writer<SimFabTestCell> *writer = new TracingWriter<SimFabTestCell>(1, 100);
        fab->addWriter(*writer);
        fab->operator()(fab->parameters());
        delete writer;
    }

    void testAddWriterToCacheBlockingSimulationFactory()
    {
        LOG(Logger::INFO, "SimulationFactoryWithCudaTest::testAddWriterToCacheBlockingSimulationFactory()")
        Writer<SimFabTestCell> *writer = new TracingWriter<SimFabTestCell>(1, 100);
        cFab->addWriter(*writer);
        cFab->operator()(cFab->parameters());
        delete writer;
    }

    void testAddWriterToCudaSimulationFactory()
    {
        LOG(Logger::INFO, "SimulationFactoryWithCudaTest::TestAddWriterToCudaSimulationFactory()")
        Writer<SimFabTestCell> *writer = new TracingWriter<SimFabTestCell>(1, 100);
        cudaFab->addWriter(*writer);
        cudaFab->operator()(cudaFab->parameters());
        delete writer;
    }

private:
    Coord<3> dim;
    unsigned maxSteps;
    SimulationFactory<SimFabTestCell> *cudaFab, *fab, *cFab;
};

}

