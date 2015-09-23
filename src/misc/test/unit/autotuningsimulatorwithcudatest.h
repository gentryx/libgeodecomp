#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/misc/simulationfactory.h>
#include <cuda.h>

#define LIBGEODECOMP_DEBUG_LEVEL 4

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

class SimulationFactoryWithCudaTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        dim = Coord<3>(100,100,100);
        maxSteps = 100;
        fab = new CudaSimulationFactory<SimFabTestCell>(SimFabTestInitializer(dim, maxSteps));
    }
    
    void tearDown()
    {
        delete fab;
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

    void testCudaSimulatiorFitness()
    {
        LOG(Logger::INFO, "SimulationFactoryWithCudaTest::testCudaSimulatorFitness()")
        for (unsigned j =6; j<=6; ++j)
            for (unsigned k = 6; k <= 6; k++)   
                for (unsigned i = 15; i <= 15; ++i)
                {
                    fab->parameters()["BlockDimX"].setValue(i);
                    fab->parameters()["BlockDimY"].setValue(k);
                    fab->parameters()["BlockDimZ"].setValue(j);
                    
                    double fitness = fab->operator()(fab->parameters());
                    LOG(Logger::INFO, j << " " << k << " " << i << " fitness: " << fitness)
                }
    }

private:
    Coord<3>dim;
    unsigned maxSteps;
    SimulationFactory<SimFabTestCell> *fab;
};

}

