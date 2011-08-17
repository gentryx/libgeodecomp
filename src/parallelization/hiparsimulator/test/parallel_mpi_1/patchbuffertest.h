#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchbuffer.h>

using namespace LibGeoDecomp;
using namespace HiParSimulator;

namespace LibGeoDecomp {
namespace HiParSimulator {

class PatchBufferTest : public CxxTest::TestSuite 
{
public:
    typedef Grid<int> GridType;
    typedef PatchBuffer<GridType, GridType> MyPatchBuffer;

    void setUp()
    {
        baseGrid = GridType(Coord<2>(7, 5), 0);
        for (int y = 0; y < 5; ++y)
            for (int x = 0; x < 7; ++x)
                baseGrid[Coord<2>(x, y)] = 10 * y + x;

        testGrid1 = GridType(Coord<2>(7, 5), 0);
        testGrid2 = GridType(Coord<2>(7, 5), 0);
        testGrid3 = GridType(Coord<2>(7, 5), 0);
        zeroGrid  = GridType(Coord<2>(7, 5), 0);
        
        // fixme: replace this by region<<coordbox
        for (int y = 0; y < baseGrid.getDimensions().y(); ++y)
            validRegion << 
                Streak<2>(Coord<2>(0, y), baseGrid.getDimensions().x());

        region0.clear();
        region1.clear();
        region1 << Streak<2>(Coord<2>(2, 2), 4);
        region1 << Streak<2>(Coord<2>(2, 3), 5);
        region2.clear();
        region2 << Streak<2>(Coord<2>(0, 0), 2);
        region2 << Streak<2>(Coord<2>(4, 1), 7);

        for (Region<2>::Iterator i = region1.begin(); i != region1.end(); ++i)
            testGrid1[*i] = i->y() * 10 + i->x();
        for (Region<2>::Iterator i = region2.begin(); i != region2.end(); ++i)
            testGrid2[*i] = i->y() * 10 + i->x();
        
        Region<2>::Iterator j = region1.begin();
        for (Region<2>::Iterator i = region2.begin(); 
             i != region2.end(); 
             ++i) {
            testGrid3[*i] = j->y() * 10 + j->x();
            ++j;
        }
    }

    void testCopyInCopyOut()
    {
        MyPatchBuffer patchBuffer;

        // check that an empty region causes no changes at all
        patchBuffer.setRegion(region0);
        patchBuffer.pushRequest(0);
        for (int i = 0; i < 4; ++i) 
            patchBuffer.put(baseGrid, validRegion, i);
        compGrid = zeroGrid;
        patchBuffer.get(compGrid, validRegion, 0);
        TS_ASSERT_EQUALS(zeroGrid, compGrid);

        // check that we can copy out regions multiple times
        patchBuffer.setRegion(region1);
        patchBuffer.pushRequest(2);
        patchBuffer.pushRequest(3);
        for (int i = 0; i < 4; ++i)
            patchBuffer.put(baseGrid, validRegion, i);
        compGrid = zeroGrid;
        patchBuffer.get(compGrid, validRegion, 2, false);
        TS_ASSERT_EQUALS(testGrid1, compGrid);
        compGrid = zeroGrid;
        patchBuffer.get(compGrid, validRegion, 2, true);
        TS_ASSERT_EQUALS(testGrid1, compGrid);

        // this is actually ugly: by changing the region, the copy out
        // targets different coordinated than the original
        // storage. but it should just work.
        patchBuffer.setRegion(region2);
        compGrid = zeroGrid;
        patchBuffer.get(compGrid, validRegion, 3);
        TS_ASSERT_EQUALS(testGrid3, compGrid);

        // just another normal retrieval
        patchBuffer.setRegion(region2);
        patchBuffer.pushRequest(1);
        for (int i = 0; i < 4; ++i)
            patchBuffer.put(baseGrid, validRegion, i);
        compGrid = zeroGrid;
        patchBuffer.get(compGrid, validRegion, 1);
        TS_ASSERT_EQUALS(testGrid2, compGrid);
    }

private:
    GridType baseGrid;
    GridType testGrid1;
    GridType testGrid2;
    GridType testGrid3;
    GridType zeroGrid;
    GridType compGrid;
    Region<2> region0;
    Region<2> region1;
    Region<2> region2;
    Region<2> validRegion;
};

}
}
