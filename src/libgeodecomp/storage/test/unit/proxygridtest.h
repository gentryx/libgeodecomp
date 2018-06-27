#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/storage/displacedgrid.h>
#include <libgeodecomp/storage/proxygrid.h>
#include <libgeodecomp/storage/soagrid.h>
#include <libgeodecomp/communication/hpxserializationwrapper.h>

/**
 * Test model for use with Boost.Serialization
 */
class MyComplicatedCell1
{
public:
    class API : public LibGeoDecomp::APITraits::HasBoostSerialization
    {};

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, int nanoStep)
    {
    }

    inline bool operator==(const MyComplicatedCell1& other)
    {
        return
            (x     == other.x) &&
            (cargo == other.cargo);
    }

    template<typename ARCHIVE>
    void serialize(ARCHIVE& archive, int version)
    {
        archive & x;
        archive & cargo;
    }

    int x;
    std::vector<int> cargo;
};

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class ProxyGridTest : public CxxTest::TestSuite
{
public:
    void testBasic()
    {
        DisplacedGrid<int> mainGrid(CoordBox<2>(Coord<2>(-1, -2), Coord<2>(22, 14)));
        ProxyGrid<int, 2> subGrid(&mainGrid, CoordBox<2>(Coord<2>(0, 0), Coord<2>(20, 10)));

        Region<2> boundingRegionSub;
        Region<2> boundingRegionMain;
        boundingRegionSub  << CoordBox<2>(Coord<2>( 0,  0), Coord<2>(20, 10));
        boundingRegionMain << CoordBox<2>(Coord<2>(-1, -2), Coord<2>(22, 14));
        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>( 0,  0), Coord<2>(20, 10)), subGrid.boundingBox());
        TS_ASSERT_EQUALS(CoordBox<2>(Coord<2>(-1, -2), Coord<2>(22, 14)), mainGrid.boundingBox());
        TS_ASSERT_EQUALS(boundingRegionSub,  subGrid.boundingRegion());
        TS_ASSERT_EQUALS(boundingRegionMain, mainGrid.boundingRegion());
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
        ProxyGrid<TestCellSoA, 3> subGrid(&mainGrid, CoordBox<3>(Coord<3>(0, 0, 0), Coord<3>(20, 10, 10)));

        std::vector<float> vecA;
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

        subGrid.loadMember(&vecA[0], MemoryLocation::HOST, s, r);
        TS_ASSERT_EQUALS(float(10.0), subGrid.get(Coord<3>( 0, 0, 0)).testValue);
        TS_ASSERT_EQUALS(float(10.1), subGrid.get(Coord<3>( 1, 0, 0)).testValue);
        TS_ASSERT_EQUALS(float(10.2), subGrid.get(Coord<3>( 2, 0, 0)).testValue);
        TS_ASSERT_EQUALS(float(10.3), subGrid.get(Coord<3>(19, 0, 0)).testValue);
        TS_ASSERT_EQUALS(float(10.4), subGrid.get(Coord<3>( 0, 9, 0)).testValue);
        TS_ASSERT_EQUALS(float(10.5), subGrid.get(Coord<3>(19, 9, 0)).testValue);
        TS_ASSERT_EQUALS(float(10.6), subGrid.get(Coord<3>( 0, 0, 9)).testValue);
        TS_ASSERT_EQUALS(float(10.7), subGrid.get(Coord<3>(19, 0, 9)).testValue);
        TS_ASSERT_EQUALS(float(10.8), subGrid.get(Coord<3>( 0, 9, 9)).testValue);
        TS_ASSERT_EQUALS(float(10.9), subGrid.get(Coord<3>(19, 9, 9)).testValue);

        TS_ASSERT_EQUALS(float(10.0), mainGrid.get(Coord<3>( 0, 0, 0)).testValue);
        TS_ASSERT_EQUALS(float(10.1), mainGrid.get(Coord<3>( 1, 0, 0)).testValue);
        TS_ASSERT_EQUALS(float(10.2), mainGrid.get(Coord<3>( 2, 0, 0)).testValue);
        TS_ASSERT_EQUALS(float(10.3), mainGrid.get(Coord<3>(19, 0, 0)).testValue);
        TS_ASSERT_EQUALS(float(10.4), mainGrid.get(Coord<3>( 0, 9, 0)).testValue);
        TS_ASSERT_EQUALS(float(10.5), mainGrid.get(Coord<3>(19, 9, 0)).testValue);
        TS_ASSERT_EQUALS(float(10.6), mainGrid.get(Coord<3>( 0, 0, 9)).testValue);
        TS_ASSERT_EQUALS(float(10.7), mainGrid.get(Coord<3>(19, 0, 9)).testValue);
        TS_ASSERT_EQUALS(float(10.8), mainGrid.get(Coord<3>( 0, 9, 9)).testValue);
        TS_ASSERT_EQUALS(float(10.9), mainGrid.get(Coord<3>(19, 9, 9)).testValue);

        std::vector<float> vecB(10);
        subGrid.saveMember(&vecB[0], MemoryLocation::HOST, s, r);
        TS_ASSERT_EQUALS(vecA, vecB);
    }

    void testLoadSaveRegion()
    {
        Coord<2> origin(200, 100);
        Coord<2> dim(50, 40);
        Coord<2> end = origin + dim;
        DisplacedGrid<TestCell<2> > mainGrid(CoordBox<2>(origin, dim));
        ProxyGrid<TestCell<2>, 2> subGrid(&mainGrid, CoordBox<2>(Coord<2>(210, 110), Coord<2>(30, 20)));

        int num = 200;
        for (int y = origin.y(); y < end.y(); y++) {
            for (int x = origin.x(); x < end.x(); x++) {
                mainGrid[Coord<2>(x, y)] =
                    TestCell<2>(Coord<2>(x, y), mainGrid.getDimensions());
                mainGrid[Coord<2>(x, y)].testValue =  num++;
            }
        }

        Region<2> region;
        region << Streak<2>(Coord<2>(210, 110), 240)
               << Streak<2>(Coord<2>(215, 111), 240)
               << Streak<2>(Coord<2>(210, 129), 230);
        std::vector<TestCell<2> > buffer(region.size());

        subGrid.saveRegion(&buffer, region);

        Region<2>::Iterator iter = region.begin();
        for (std::size_t i = 0; i < region.size(); ++i) {
            TestCell<2> actual = mainGrid.get(*iter);
            TestCell<2> expected(*iter, mainGrid.getDimensions());
            int expectedIndex = 200 + (*iter - origin).toIndex(dim);
            expected.testValue = expectedIndex;

            TS_ASSERT_EQUALS(actual, expected);
            ++iter;
        }

        // manipulate test data:
        for (std::size_t i = 0; i < region.size(); ++i) {
            buffer[i].pos = Coord<2>(-i, -10);
        }

        int index = 0;
        subGrid.loadRegion(buffer, region);
        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i) {
            Coord<2> actual = mainGrid.get(*i).pos;
            Coord<2> expected = Coord<2>(index, -10);
            TS_ASSERT_EQUALS(actual, expected);

            --index;
        }
    }

    void testLoadSaveRegionWithBoostSerialization()
    {
#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
        typedef DisplacedGrid<MyComplicatedCell1> GridType;

        Coord<2> dim(300, 400);
        CoordBox<2> box(Coord<2>(-100, -100), dim);

        GridType mainSendGrid(box);
        GridType mainRecvGrid(box);

        ProxyGrid<MyComplicatedCell1, 2> subSendGrid(&mainSendGrid, CoordBox<2>(Coord<2>(-20, -20), Coord<2>(50, 60)));
        ProxyGrid<MyComplicatedCell1, 2> subRecvGrid(&mainRecvGrid, CoordBox<2>(Coord<2>(-20, -20), Coord<2>(50, 60)));

        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            MyComplicatedCell1 cell;
            cell.cargo << i->x();
            cell.cargo << i->y();
            cell.x = i->x() * 100 + i->y();
            mainSendGrid.set(*i, cell);
        }

        Region<2> region;
        region << Streak<2>(Coord<2>(5001, 7001), 5012)
               << Streak<2>(Coord<2>(5001, 7002), 5002)
               << Streak<2>(Coord<2>(5011, 7002), 5012)
               << Streak<2>(Coord<2>(5001, 7003), 5002)
               << Streak<2>(Coord<2>(5011, 7003), 5012)
               << Streak<2>(Coord<2>(5001, 7004), 5002)
               << Streak<2>(Coord<2>(5011, 7004), 5012)
               << Streak<2>(Coord<2>(5001, 7005), 5002)
               << Streak<2>(Coord<2>(5011, 7005), 5012)
               << Streak<2>(Coord<2>(5001, 7006), 5002)
               << Streak<2>(Coord<2>(5011, 7006), 5012)
               << Streak<2>(Coord<2>(5001, 7007), 5002)
               << Streak<2>(Coord<2>(5011, 7007), 5012)
               << Streak<2>(Coord<2>(5001, 7008), 5012);
        TS_ASSERT_EQUALS(region.size(), 34);
        Coord<2> offset(-5000, -7000);

        std::vector<char> buffer;
        subSendGrid.saveRegion(&buffer, region, offset);

        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i) {
            TS_ASSERT_DIFFERS(mainSendGrid[*i + offset], mainRecvGrid[*i + offset]);
        }
        subRecvGrid.loadRegion(buffer, region, offset);
        for (Region<2>::Iterator i = region.begin(); i != region.end(); ++i) {
            TS_ASSERT_EQUALS(mainSendGrid[*i + offset], mainRecvGrid[*i + offset]);
        }
#endif
    }
};

}
