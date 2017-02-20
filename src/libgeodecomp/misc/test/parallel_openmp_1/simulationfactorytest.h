#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/misc/simulationfactory.h>
#include <cuda.h>
#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

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
    void update(const COORD_MAP& neighborhood, unsigned nanoStep)
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
    void updateCuda(const COORD_MAP& neighborhood, unsigned nanoStep)
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
    SimFabTestInitializer(Coord<3> gridDimensions, unsigned timeSteps) :
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

class SimulationFactoryTest : public CxxTest::TestSuite
{
public:

    void testBasic()
    {
    //    Coord<3> dim(100, 100, 100);
    //    int maxSteps = 250;

    //    CacheBlockingSimulationFactory<SimFabTestCell> fab(SimFabTestInitializer(dim, maxSteps));
        //fab.parameters()["Simulator"] = "CacheBlockingSimulator";

    //    Simulator<SimFabTestCell> *sim = fab();
    //    sim->run();
    //    delete sim;
    }

    void testFitness()
    {
    //    SimulationParameters params;
    //    std::vector<std::string> types;
    //    types << "SerialSimulator";
        //params.addParameter("Simulator", types);

    //    Coord<3> dim(50, 50, 50);
    //    int maxSteps = 400;
    //    SerialSimulationFactory<SimFabTestCell> fab(SimFabTestInitializer(dim, maxSteps));
    //    double fitness = fab(params);
    //    std::cout << "fitness: " << fitness << "\n";
    }
};

}
