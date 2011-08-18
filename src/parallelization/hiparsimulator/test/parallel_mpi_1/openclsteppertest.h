#include <cxxtest/TestSuite.h>

#include <libgeodecomp/config.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/parallelization/hiparsimulator/stepperhelper.h>
#include <libgeodecomp/parallelization/hiparsimulator/openclstepper.h>

using namespace LibGeoDecomp; 
using namespace HiParSimulator; 

namespace LibGeoDecomp {
namespace HiParSimulator {

class JacobiCell
{
public:
    typedef Topologies::Cube<3>::Topology Topology;

    static inline unsigned nanoSteps() { return 1; }
    
    JacobiCell(const double& _temp=-1) :
        temp(_temp)
    {}

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighbors, const unsigned& nanoStep) 
    {
        temp = (neighbors[Coord<3>( 0,  0, -1)] +
                neighbors[Coord<3>( 0, -1,  0)] +
                neighbors[Coord<3>(-1,  0,  0)] +
                neighbors[Coord<3>( 0,  1,  0)] +
                neighbors[Coord<3>( 0,  0,  1)]) * (1.0 / 6.0);
    }

private:
    double temp;
};

class JacobiInitializer : public SimpleInitializer<JacobiCell>
{
public:
    JacobiInitializer(Coord<3> dimensions) :
        SimpleInitializer(dimensions)
    {}

    virtual void grid(GridBase<JacobiCell, 3> *ret)
    {
        // fixme: andi, implement me!
    }
};

class OpenCLStepperBasicTest : public CxxTest::TestSuite
{
public:
    typedef OpenCLStepper<JacobiCell> MyStepper;

    void setUp()
    {
        init.reset(new JacobiInitializer(Coord<3>(128, 128, 128)));
        CoordBox<3> rect = init->gridBox();

        partitionManager.reset(new PartitionManager<3>(rect));
        stepper.reset(
            new MyStepper(partitionManager, init));
    }

    void testBasic()
    {
        std::cout << "fixme: andi, implement me!\n";
    }

private:
    boost::shared_ptr<JacobiInitializer> init;
    boost::shared_ptr<PartitionManager<3> > partitionManager;
    boost::shared_ptr<MyStepper> stepper;
};

}
}
