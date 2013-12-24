#include <cxxtest/TestSuite.h>

#include <libgeodecomp/geometry/partitions/recursivebisectionpartition.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/hiparsimulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/cudastepper.h>

using namespace LibGeoDecomp;
using namespace HiParSimulator;

namespace LibGeoDecomp {
namespace HiParSimulator {

class CUDAStepperBasicTest : public CxxTest::TestSuite
{
public:
    void testBasic()
    {
        HiParSimulator<TestCell<2>, RecursiveBisectionPartition<2>, CUDAStepper<TestCell<2> > > sim(
            new TestInitializer<TestCell<2> >(Coord<2>(17, 12)));
        sim.run();
    }
};

}
}
