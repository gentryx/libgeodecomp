#include <cxxtest/TestSuite.h>

#include <libgeodecomp/config.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/parallelization/hiparsimulator/stepperhelper.h>
#include <libgeodecomp/parallelization/hiparsimulator/openclstepper.h>
#include <libgeodecomp/parallelization/hiparsimulator/test/parallel_mpi_1/cell.h>

using namespace LibGeoDecomp; 
using namespace HiParSimulator; 

namespace LibGeoDecomp {
namespace HiParSimulator {

#ifdef LIBGEODECOMP_FEATURE_OPENCL
class CellInitializer : public SimpleInitializer<Cell>
{
public:
    CellInitializer(Coord<3> dimensions) :
        SimpleInitializer<Cell>(dimensions)
    {}

    virtual void grid(GridBase<Cell, 3> *ret)
    {
        // fixme: andi, implement me!
    }
};
#endif

class OpenCLStepperBasicTest : public CxxTest::TestSuite
{
public:
#ifdef LIBGEODECOMP_FEATURE_OPENCL
    typedef OpenCLStepper<Cell> MyStepper;

    void setUp()
    {
        init.reset(new JacobiInitializer(Coord<3>(128, 128, 128)));
        CoordBox<3> rect = init->gridBox();

        partitionManager.reset(new PartitionManager<3>(rect));
        stepper.reset(
            new MyStepper(cellSourceFile, partitionManager, init));
    }
#endif

    void testBasic()
    {
#ifdef LIBGEODECOMP_FEATURE_OPENCL
        std::cout << "fixme: andi, implement me!\n";
#endif
    }

#ifdef LIBGEODECOMP_FEATURE_OPENCL
private:
    boost::shared_ptr<JacobiInitializer> init;
    boost::shared_ptr<PartitionManager<3> > partitionManager;
    boost::shared_ptr<MyStepper> stepper;
#endif
};

}
}
