#include <cxxtest/TestSuite.h>
#include <libgeodecomp/storage/displacedgrid.h>
#include <libgeodecomp/storage/proxygrid.h>
#include <libgeodecomp/storage/soagrid.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class ProxyGridTest : public CxxTest::TestSuite
{
public:
    void testBasic()
    {
        std::cout << "============================================================================\n";
        DisplacedGrid<int> mainGrid(CoordBox<2>(Coord<2>(-1, -2), Coord<2>(22, 14)));
        ProxyGrid<int, 2> subGrid(mainGrid, CoordBox<2>(Coord<2>(0, 0), Coord<2>(20, 10)));

        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>( 0,  0), Coord<2>(20, 10)), subGrid.boundingBox());
        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>(-1, -2), Coord<2>(22, 14)), mainGrid.boundingBox());
        subGrid.setEdge(47);
        TS_ASSERT_EQUALS(47, subGrid.getEdge());
        TS_ASSERT_EQUALS(47, mainGrid.getEdge());

        subGrid.set(Coord<2>(12, 6), 11);
        TS_ASSERT_EQUALS(11, subGrid.get(Coord<2>(12, 6)));
        TS_ASSERT_EQUALS(11, mainGrid.get(Coord<2>(12, 6)));

        Streak<2> streak(Coord<2>(2, 9), 8);
        std::vector<int> vecA;
        vecA << 1
             << 1
             << 2
             << 3
             << 5
             << 8;
        subGrid.set(streak, &vecA[0]);

        TS_ASSERT_EQUALS(1, subGrid.get(Coord<2>(2, 9)));
        TS_ASSERT_EQUALS(1, subGrid.get(Coord<2>(3, 9)));
        TS_ASSERT_EQUALS(2, subGrid.get(Coord<2>(4, 9)));
        TS_ASSERT_EQUALS(3, subGrid.get(Coord<2>(5, 9)));
        TS_ASSERT_EQUALS(5, subGrid.get(Coord<2>(6, 9)));
        TS_ASSERT_EQUALS(8, subGrid.get(Coord<2>(7, 9)));

        TS_ASSERT_EQUALS(1, mainGrid.get(Coord<2>(2, 9)));
        TS_ASSERT_EQUALS(1, mainGrid.get(Coord<2>(3, 9)));
        TS_ASSERT_EQUALS(2, mainGrid.get(Coord<2>(4, 9)));
        TS_ASSERT_EQUALS(3, mainGrid.get(Coord<2>(5, 9)));
        TS_ASSERT_EQUALS(5, mainGrid.get(Coord<2>(6, 9)));
        TS_ASSERT_EQUALS(8, mainGrid.get(Coord<2>(7, 9)));

        std::vector<int> vecB(6);
        std::vector<int> vecC(6);
        TS_ASSERT_DIFFERS(vecA, vecB);
        TS_ASSERT_DIFFERS(vecA, vecC);

        subGrid.get(streak, &vecB[0]);
        mainGrid.get(streak, &vecC[0]);

        TS_ASSERT_EQUALS(vecA, vecB);
        TS_ASSERT_EQUALS(vecA, vecC);
    }

    void testSelector()
    {
        SoAGrid<TestCellSoA, Topologies::Cube<3>::Topology> mainGrid(
            CoordBox<3>(Coord<3>(-3, -2, -1), Coord<3>(26, 14, 12)));
        ProxyGrid<TestCellSoA, 3> subGrid(mainGrid, CoordBox<3>(Coord<3>(0, 0, 0), Coord<3>(20, 10, 10)));

        std::vector<double> vecA;
        vecA << 10.0
             << 10.1
             << 10.2
             << 10.3
             << 10.4
             << 10.5
             << 10.6
             << 10.7
             << 10.8
             << 10.9;

        Selector<TestCellSoA> s(&TestCellSoA::testValue, "we don't really need a name here...");
        Region<3> r;
        r << Coord<3>( 0, 0, 0)
          << Coord<3>( 1, 0, 0)
          << Coord<3>( 2, 0, 0)
          << Coord<3>(19, 0, 0)
          << Coord<3>( 0, 9, 0)
          << Coord<3>(19, 9, 0)
          << Coord<3>( 0, 0, 9)
          << Coord<3>(19, 0, 9)
          << Coord<3>( 0, 9, 9)
          << Coord<3>(19, 9, 9);

        subGrid.loadMember(&vecA[0], s, r);
        TS_ASSERT_EQUALS(10.0, subGrid.get(Coord<3>( 0, 0, 0)).testValue);
        TS_ASSERT_EQUALS(10.1, subGrid.get(Coord<3>( 1, 0, 0)).testValue);
        TS_ASSERT_EQUALS(10.2, subGrid.get(Coord<3>( 2, 0, 0)).testValue);
        TS_ASSERT_EQUALS(10.3, subGrid.get(Coord<3>(19, 0, 0)).testValue);
        TS_ASSERT_EQUALS(10.4, subGrid.get(Coord<3>( 0, 9, 0)).testValue);
        TS_ASSERT_EQUALS(10.5, subGrid.get(Coord<3>(19, 9, 0)).testValue);
        TS_ASSERT_EQUALS(10.6, subGrid.get(Coord<3>( 0, 0, 9)).testValue);
        TS_ASSERT_EQUALS(10.7, subGrid.get(Coord<3>(19, 0, 9)).testValue);
        TS_ASSERT_EQUALS(10.8, subGrid.get(Coord<3>( 0, 9, 9)).testValue);
        TS_ASSERT_EQUALS(10.9, subGrid.get(Coord<3>(19, 9, 9)).testValue);

        TS_ASSERT_EQUALS(10.0, mainGrid.get(Coord<3>( 0, 0, 0)).testValue);
        TS_ASSERT_EQUALS(10.1, mainGrid.get(Coord<3>( 1, 0, 0)).testValue);
        TS_ASSERT_EQUALS(10.2, mainGrid.get(Coord<3>( 2, 0, 0)).testValue);
        TS_ASSERT_EQUALS(10.3, mainGrid.get(Coord<3>(19, 0, 0)).testValue);
        TS_ASSERT_EQUALS(10.4, mainGrid.get(Coord<3>( 0, 9, 0)).testValue);
        TS_ASSERT_EQUALS(10.5, mainGrid.get(Coord<3>(19, 9, 0)).testValue);
        TS_ASSERT_EQUALS(10.6, mainGrid.get(Coord<3>( 0, 0, 9)).testValue);
        TS_ASSERT_EQUALS(10.7, mainGrid.get(Coord<3>(19, 0, 9)).testValue);
        TS_ASSERT_EQUALS(10.8, mainGrid.get(Coord<3>( 0, 9, 9)).testValue);
        TS_ASSERT_EQUALS(10.9, mainGrid.get(Coord<3>(19, 9, 9)).testValue);

        std::vector<double> vecB(10);
        subGrid.saveMember(&vecB[0], s, r);
        TS_ASSERT_EQUALS(vecA, vecB);
}
};

}
