#include <cuda.h>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/parallelization/cudasimulator.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CudaSimulatorTest : public CxxTest::TestSuite
{
public:
    typedef TestCell<3, Stencils::VonNeumann<3, 1>, Topologies::Cube<3>::Topology,
                     TestCellHelpers::EmptyAPI, TestCellHelpers::NoOutput> TestCell3D;
    typedef TestInitializer<TestCell3D> TestInitializer3D;

    void testBasic()
    {
        std::cout << "----------------------------------------------\n";
        CudaSimulator<TestCell3D> sim(new TestInitializer3D());
        std::cout << "----------------------------------------------\n";
    }
};

}
