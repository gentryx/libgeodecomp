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
    typedef PatchBuffer<GridType, GridType, int> MyPatchBuffer;

    void setUp()
    {
        baseGrid = GridType(Coord<2>(7, 5), 0);
        for (int y = 0; y < 5; ++y)
            for (int x = 0; x < 7; ++x)
                baseGrid[Coord<2>(x, y)] = 10 * y + x;

        testGrid1 = GridType(Coord<2>(7, 5), 0);
        testGrid2 = GridType(Coord<2>(7, 5), 0);
        zeroGrid  = GridType(Coord<2>(7, 5));
        
        // fixme: replace this by region<<coordbox
        for (int y = 0; y < baseGrid.getDimensions().y(); ++y)
            validRegion << 
                Streak<2>(Coord<2>(0, y), baseGrid.getDimensions().x());

        patchBuffer = MyPatchBuffer();

        region0.clear();
        region1.clear();
        region1 << Streak<2>(Coord<2>(2, 2), 4);
        region1 << Streak<2>(Coord<2>(2, 3), 5);
        region2.clear();
        region2 << Streak<2>(Coord<2>(0, 0), 7);
        region2 << Streak<2>(Coord<2>(4, 1), 7);

        for (Region<2>::Iterator i = region1.begin(); i != region1.end(); ++i)
            testGrid1[*i] = i->y() * 10 + i->x();
        for (Region<2>::Iterator i = region2.begin(); i != region2.end(); ++i)
            testGrid2[*i] = i->y() * 10 + i->x();
    }

    void testCopyInCopyOut()
    {
        patchBuffer.pushRequest(&region0, 0);
        patchBuffer.pushRequest(&region1, 2);
        patchBuffer.pushRequest(&region2, 3);

        for (int i = 0; i < 4; ++i)
            patchBuffer.put(baseGrid, validRegion, i);
        
        compGrid = zeroGrid;
        patchBuffer.get(compGrid, region0, 0);
        TS_ASSERT_EQUALS(zeroGrid, compGrid);

        compGrid = zeroGrid;
        patchBuffer.get(compGrid, region1, 2);
        TS_ASSERT_EQUALS(testGrid1, compGrid);

        patchBuffer.pushRequest(&region1, 9);

        for (int i = 4; i < 14; ++i)
            patchBuffer.put(baseGrid, validRegion, i);

        compGrid = zeroGrid;
        patchBuffer.get(compGrid, region2, 3);
        TS_ASSERT_EQUALS(testGrid2, compGrid);

        compGrid = zeroGrid;
        patchBuffer.get(compGrid, region1, 9);
        TS_ASSERT_EQUALS(testGrid1, compGrid);
    }

    GridType baseGrid;
    GridType testGrid1;
    GridType testGrid2;
    GridType zeroGrid;
    GridType compGrid;
    Region<2> region0;
    Region<2> region1;
    Region<2> region2;
    Region<2> validRegion;
    MyPatchBuffer patchBuffer;
};

}
}
