#include <cxxtest/TestSuite.h>

#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/parallelization/hiparsimulator/patchlink.h>

using namespace LibGeoDecomp; 
using namespace HiParSimulator; 

namespace LibGeoDecomp {
namespace HiParSimulator {

class PatchLinkTest : public CxxTest::TestSuite
{
public:
    typedef Grid<int> GridType;

    void setUp()
    {
        region1.clear();
        region1 << Streak<2>(Coord<2>(2, 2), 4);
        region1 << Streak<2>(Coord<2>(2, 3), 5);
        region2.clear();
        region2 << Streak<2>(Coord<2>(0, 0), 6);
        region2 << Streak<2>(Coord<2>(4, 1), 5);

        zeroGrid  = GridType(Coord<2>(7, 5), 0);
        boundingBox.clear();
        boundingBox << CoordBox<2>(Coord<2>(0, 0), Coord<2>(7, 5));
            
        sendGrid1 = markGrid(region1, mpiLayer.rank());
        sendGrid2 = markGrid(region2, mpiLayer.rank());

        tag = 69;
    }

    void testBasic() 
    {
        int nanoStep = 0;
        for (int sender = 0; sender < mpiLayer.size(); ++sender) {
            for (int receiver = 0; receiver < mpiLayer.size(); ++receiver) {
                if (sender != receiver) {
                    Region<2>& region  = sender % 2 ? region2 : region1;
                    GridType& sendGrid = sender % 2 ? sendGrid2 : sendGrid1;
                    long nanoStep = sender * receiver + 4711;
                    
                    if (sender == mpiLayer.rank()) {
                        acc = PatchLink<GridType>::Accepter(
                            &mpiLayer,
                            region,
                            tag,
                            receiver);

                        acc.pushRequest(nanoStep);
                        acc.put(sendGrid, boundingBox, nanoStep);
                        acc.wait();
                    }

                    if (receiver == mpiLayer.rank()) {
                        pro = PatchLink<GridType>::Provider(
                            &mpiLayer,
                            region,
                            tag,
                            sender);

                        expected = markGrid(region, sender);
                        actual = zeroGrid;

                        pro.recv(nanoStep);
                        pro.wait();
                        // pro.get(actual, boundingBox, nanoStep);

                        // fixme: TS_ASSERT_EQUALS(actual, expected);
                    }
                }

                ++nanoStep;
            }
        }
    }

private:
    int tag;
    MPILayer mpiLayer;

    GridType zeroGrid;
    GridType sendGrid1;
    GridType sendGrid2;
    GridType expected;
    GridType actual;

    Region<2> boundingBox;
    Region<2> region1;
    Region<2> region2;

    PatchLink<GridType>::Accepter acc;
    PatchLink<GridType>::Provider pro;

    GridType markGrid(const Region<2>& region, const int& rank)
    {
        GridType ret = zeroGrid;
        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i)
            ret[*i] = rank + i->y() * 10 + i->x();
        return ret;
    }
};

}
}
