#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/io/varstepinitializerproxy.h>
#include <libgeodecomp/misc/simfabtestmodel.h>
#include <libgeodecomp/misc/cacheblockingsimulationfactory.h>
#include <libgeodecomp/misc/cudasimulationfactory.h>
#include <libgeodecomp/misc/serialsimulationfactory.h>
#include <libgeodecomp/misc/simulationfactory.h>
#include <cuda.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

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
            LOG(Logger::DBG, "setMaxSteps("<<i<<")");
            initializerProxy->setMaxSteps(i);
            LOG(Logger::DBG,"i: "<< i << " maxSteps(): "
                << initializerProxy->maxSteps());
            TS_ASSERT_EQUALS(i, initializerProxy->maxSteps());
            aktFitness = fab->operator()(fab->parameterSet);
            LOG(Logger::DBG, "Fitness: " << aktFitness);
            TS_ASSERT(oldFitness > aktFitness);
            oldFitness = aktFitness;
        }

        LOG(Logger::DBG, "getInitializer()->maxSteps(): "
            << initializerProxy->proxyObj->maxSteps()
            << " \"initial\" maxSteps: " << maxSteps);
        TS_ASSERT_EQUALS(initializerProxy->proxyObj->maxSteps(), maxSteps);
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
            cFab->parameterSet["PipelineLength"].setValue(1);
            cFab->parameterSet["WavefrontWidth"].setValue(100);
            cFab->parameterSet["WavefrontHeight"].setValue(40);
            double fitness = cFab->operator()(cFab->parameterSet);
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
            cudaFab->parameterSet["BlockDimX"].setValue(15);
            cudaFab->parameterSet["BlockDimY"].setValue(6);
            cudaFab->parameterSet["BlockDimZ"].setValue(6);
            double fitness = cudaFab->operator()(cudaFab->parameterSet);
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
        double fitness = cFab->operator()(cFab->parameterSet);
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
        fab->operator()(fab->parameterSet);
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
        cFab->operator()(cFab->parameterSet);
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
        cudaFab->operator()(cudaFab->parameterSet);
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
