#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/nonpodtestcell.h>
#include <libgeodecomp/parallelization/serialsimulator.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class NonPoDTestCellTest : public CxxTest::TestSuite
{
public:
    void testBasic()
    {
        typedef NonPoDTestCell::Initializer Initializer;
        SerialSimulator<NonPoDTestCell> sim(new Initializer());
        sim.run();
        NonPoDTestCell cell = sim.getGrid()->get(Coord<2>());

        std::set<Coord<2> > expectedNeighbors;
        for (int y = 0; y < 10; ++y) {
            for (int x = 0; x < 15; ++x) {
                expectedNeighbors << Coord<2>(x, y);
            }
        }

        TS_ASSERT_EQUALS(cell.seenNeighbors, expectedNeighbors);
        TS_ASSERT(cell.missingNeighbors.empty());
    }
};

}
