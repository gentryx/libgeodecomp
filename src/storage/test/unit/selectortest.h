#include <libgeodecomp/misc/color.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/storage/grid.h>
#include <libgeodecomp/storage/selector.h>
#include <libgeodecomp/storage/simplefilter.h>
#include <libgeodecomp/storage/soagrid.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class MyDummyCell
{
public:
    class API :
        public APITraits::HasSoA
    {};

    explicit MyDummyCell(const int x = 0, const double y = 0, const char z = 0) :
        x(x),
        y(y),
        z(z)
    {}

    long long x;
    double y;
    char z;
};

class MyOtherDummyCell
{
public:
    class API :
        public APITraits::HasSoA
    {};

    explicit MyOtherDummyCell(
        const int x = 0,
        const double y1 = 0,
        const double y2 = 0,
        const double y3 = 0) :
        x(x)
    {
        y[0] = y1;
        y[1] = y2;
        y[2] = y3;
    }

    int x;
    double y[3];
};

}

LIBFLATARRAY_REGISTER_SOA(LibGeoDecomp::MyDummyCell,      ((long long)(x))((double)(y))((char)(z)) )
LIBFLATARRAY_REGISTER_SOA(LibGeoDecomp::MyOtherDummyCell, ((int)(x))((double)(y)(3)) )

namespace LibGeoDecomp {

class SelectorTest : public CxxTest::TestSuite
{
public:
    void testBasic()
    {
        Selector<MyDummyCell> selectorX(&MyDummyCell::x, "varX");
        Selector<MyDummyCell> selectorY(&MyDummyCell::y, "varY");
        Selector<MyDummyCell> selectorZ(&MyDummyCell::z, "varZ");

        TS_ASSERT_EQUALS("varX", selectorX.name());
        TS_ASSERT_EQUALS("varY", selectorY.name());
        TS_ASSERT_EQUALS("varZ", selectorZ.name());

        TS_ASSERT_EQUALS(sizeof(long long), selectorX.sizeOfMember());
        TS_ASSERT_EQUALS(sizeof(double),    selectorY.sizeOfMember());
        TS_ASSERT_EQUALS(sizeof(char),      selectorZ.sizeOfMember());

        TS_ASSERT_EQUALS(sizeof(long long), selectorX.sizeOfExternal());
        TS_ASSERT_EQUALS(sizeof(double),    selectorY.sizeOfExternal());
        TS_ASSERT_EQUALS(sizeof(char),      selectorZ.sizeOfExternal());

        TS_ASSERT_EQUALS(0,                                       selectorX.offset());
        TS_ASSERT_EQUALS(int(sizeof(long long)),                  selectorY.offset());
        TS_ASSERT_EQUALS(int(sizeof(long long) + sizeof(double)), selectorZ.offset());

        TS_ASSERT_EQUALS(1, selectorX.arity());
        TS_ASSERT_EQUALS(1, selectorY.arity());
        TS_ASSERT_EQUALS(1, selectorZ.arity());

        TS_ASSERT_THROWS(selectorX.typeName(), std::invalid_argument);
        TS_ASSERT_EQUALS("DOUBLE", selectorY.typeName());
        TS_ASSERT_EQUALS("BYTE",   selectorZ.typeName());
    }

    void testCopyMemberOut()
    {
        Selector<MyDummyCell> selectorX(&MyDummyCell::x, "varX");
        Selector<MyDummyCell> selectorY(&MyDummyCell::y, "varY");
        Selector<MyDummyCell> selectorZ(&MyDummyCell::z, "varZ");

        std::vector<MyDummyCell> vec;
        for (int i = 0; i < 20; ++i) {
            vec << MyDummyCell(i, 47.11 + i, 'a' + i);
        }

        std::vector<long long>    targetX(20, -1);
        std::vector<double> targetY(20, -1);
        std::vector<char>   targetZ(20, 'A');

        selectorX.copyMemberOut(&vec[0], (char*)&targetX[0], 20);
        selectorY.copyMemberOut(&vec[0], (char*)&targetY[0], 20);
        selectorZ.copyMemberOut(&vec[0], (char*)&targetZ[0], 20);

        for (int i = 0; i < 20; ++i) {
            TS_ASSERT_EQUALS(targetX[i], i);
            TS_ASSERT_EQUALS(targetY[i], 47.11 + i);
            TS_ASSERT_EQUALS(targetZ[i], 'a' + i);
        }
    }

    void testCopyMemberIn()
    {
        Selector<MyDummyCell> selectorX(&MyDummyCell::x, "varX");
        Selector<MyDummyCell> selectorY(&MyDummyCell::y, "varY");
        Selector<MyDummyCell> selectorZ(&MyDummyCell::z, "varZ");

        std::vector<MyDummyCell> vec(20);
        std::vector<long long> targetX;
        std::vector<double>    targetY;
        std::vector<char>      targetZ;

        for (int i = 0; i < 20; ++i) {
            targetX << i * 2 + 13;
            targetY << 1.0 + i / 100.0;
            targetZ << i + 69;
        }

        selectorX.copyMemberIn((char*)&targetX[0], &vec[0], 20);
        selectorY.copyMemberIn((char*)&targetY[0], &vec[0], 20);
        selectorZ.copyMemberIn((char*)&targetZ[0], &vec[0], 20);

        for (int i = 0; i < 20; ++i) {
            TS_ASSERT_EQUALS(targetX[i], i * 2 + 13);
            TS_ASSERT_EQUALS(targetY[i], 1.0 + i / 100.0);
            TS_ASSERT_EQUALS(targetZ[i], i + 69);
        }
    }

    class MyDummyFilter : public Filter<MyDummyCell, double, Color>
    {
    public:
        void copyStreakInImpl(const Color *source, double *target, const std::size_t num, const std::size_t stride)
        {
            for (std::size_t i = 0; i < num; ++i) {
                target[i] = source[i].red() * 2 + 10;
            }
        }

        void copyStreakOutImpl(const double *source, Color *target, const std::size_t num, const std::size_t stride)
        {
            for (std::size_t i = 0; i < num; ++i) {
                target[i] = Color(source[i], 47, 11);
            }
        }

        void copyMemberInImpl(
            const Color *source, MyDummyCell *target, std::size_t num, double MyDummyCell:: *memberPointer)
        {
            for (std::size_t i = 0; i < num; ++i) {
                target[i].*memberPointer = source[i].red() * 2 + 10;
            }
        }

        void copyMemberOutImpl(
            const MyDummyCell *source, Color *target, std::size_t num, double MyDummyCell:: *memberPointer)
        {
            for (std::size_t i = 0; i < num; ++i) {
                target[i] = Color(source[i].*memberPointer, 47, 11);
            }
        }
    };

    class MySimpleFilter : public SimpleFilter<MyDummyCell, char, double>
    {
    public:
        void load(const double& source, char *target)
        {
            *target = source + 10;
        }

        void save(const char& source, double *target)
        {
            *target = source + 20;
        }
    };

    void testLocalFilter()
    {
        class FancyFilter : public SimpleFilter<MyDummyCell, char, double>
        {
        public:
            void load(const double& source, char *target)
            {
                *target = source + 11;
            }

            void save(const char& source, double *target)
            {
                *target = source + 21;
            }
        };

        FilterBase<MyDummyCell> *filter1 = new FancyFilter();
        boost::shared_ptr<FilterBase<MyDummyCell> > filter2(filter1);
        Selector<MyDummyCell> selector(&MyDummyCell::z, "varZ", filter2);

        std::vector<MyDummyCell> vec;
        for (int i = 0; i < 13; ++i) {
            vec << MyDummyCell(i, i, 'a' + i);
        }

        std::vector<double> target(13);
        selector.copyMemberOut(&vec[0], (char*)&target[0], 13);
        for (int i = 0; i < 13; ++i) {
            // expecting 'a' + offset from save() + index
            double expected = 97 + 21 + i;
            TS_ASSERT_EQUALS(expected, target[i]);
        }
    }

    void testFilterAoS1()
    {
        // test copyMemberOut:
        boost::shared_ptr<FilterBase<MyDummyCell> > filter(
            new MyDummyFilter());
        Selector<MyDummyCell> selectorY(&MyDummyCell::y, "varY", filter);

        selectorY.checkTypeID<Color>();

        std::vector<MyDummyCell> vec;
        for (int i = 0; i < 20; ++i) {
            vec << MyDummyCell(i, i, 'a' + i);
        }

        std::vector<Color> targetY(20);
        selectorY.copyMemberOut(&vec[0], (char*)&targetY[0], 20);

        for (int i = 0; i < 20; ++i) {
            TS_ASSERT_EQUALS(Color(i, 47, 11), targetY[i]);
        }

        // test copyMemberIn:
        selectorY.copyMemberIn((char*)&targetY[0], &vec[0], 20);

        for (int i = 0; i < 20; ++i) {
            TS_ASSERT_EQUALS(vec[i].y, i * 2 + 10);
        }
    }

    void testFilterAoS2()
    {
        // test copyMemberOut:
        boost::shared_ptr<FilterBase<MyDummyCell> > filter(
            new MySimpleFilter());
        Selector<MyDummyCell> selectorZ(&MyDummyCell::z, "varZ", filter);

        selectorZ.checkTypeID<double>();

        std::vector<MyDummyCell> vec;
        for (int i = 0; i < 20; ++i) {
            vec << MyDummyCell(i, i, 'a' + i);
        }

        std::vector<double> targetZ(20);
        selectorZ.copyMemberOut(&vec[0], (char*)&targetZ[0], 20);

        for (int i = 0; i < 20; ++i) {
            TS_ASSERT_EQUALS(i + 20 + 'a', targetZ[i]);
        }

        // test copyMemberIn:
        selectorZ.copyMemberIn((char*)&targetZ[0], &vec[0], 20);

        for (int i = 0; i < 20; ++i) {
            char expected = i;
            expected += 30 + 'a';
            TS_ASSERT_EQUALS((int)vec[i].z, expected);
        }
    }

    void testFilterSoA1()
    {
        // test copyStreakOut:
        boost::shared_ptr<FilterBase<MyDummyCell> > filter(
            new MyDummyFilter());
        Selector<MyDummyCell> selectorY(&MyDummyCell::y, "varY", filter);

        selectorY.checkTypeID<Color>();

        std::vector<double> vec;
        for (int i = 0; i < 20; ++i) {
            vec << i + 50;
        }

        std::vector<Color> targetY(20);
        selectorY.copyStreakOut((char*)&vec[0], (char*)&targetY[0], 20, 0);

        for (int i = 0; i < 20; ++i) {
            TS_ASSERT_EQUALS(Color(i + 50, 47, 11), targetY[i]);
        }

        // test copyStreakIn:
        selectorY.copyStreakIn((char*)&targetY[0], (char*)&vec[0], 20, 0);

        for (int i = 0; i < 20; ++i) {
            TS_ASSERT_EQUALS(vec[i], (50 + i) * 2 + 10);
        }
    }

    void testFilterSoA2()
    {
        // test copyStreakOut:
        boost::shared_ptr<FilterBase<MyDummyCell> > filter(
            new MyDummyFilter());
        Selector<MyDummyCell> selectorY(&MyDummyCell::y, "varY", filter);

        TS_ASSERT_EQUALS(sizeof(double), selectorY.sizeOfMember());
        TS_ASSERT_EQUALS(sizeof(Color),  selectorY.sizeOfExternal());

        selectorY.checkTypeID<Color>();

        CoordBox<2> box(Coord<2>(), Coord<2>(20, 1));
        Region<2> region;
        region << box;
        SoAGrid<MyDummyCell> grid(box);

        for (int i = 0; i < 20; ++i) {
            grid.set(Coord<2>(i, 0), MyDummyCell(4711, i + 50));
        }

        std::vector<Color> targetY(20);
        grid.saveMember(&targetY[0], selectorY, region);

        for (int i = 0; i < 20; ++i) {
            TS_ASSERT_EQUALS(Color(i + 50, 47, 11), targetY[i]);
        }

        // test copyStreakIn:
        grid.loadMember(&targetY[0], selectorY, region);

        for (int i = 0; i < 20; ++i) {
            TS_ASSERT_EQUALS(grid.get(Coord<2>(i, 0)).y, (50 + i) * 2 + 10);
        }
    }

    void testFilterSoA3()
    {
        // test copyStreakOut:
        boost::shared_ptr<FilterBase<MyDummyCell> > filter(
            new MySimpleFilter());
        Selector<MyDummyCell> selectorZ(&MyDummyCell::z, "varZ", filter);

        TS_ASSERT_EQUALS(sizeof(char),   selectorZ.sizeOfMember());
        TS_ASSERT_EQUALS(sizeof(double), selectorZ.sizeOfExternal());

        selectorZ.checkTypeID<double>();

        CoordBox<2> box(Coord<2>(), Coord<2>(20, 1));
        Region<2> region;
        region << box;
        SoAGrid<MyDummyCell> grid(box);

        for (int i = 0; i < 20; ++i) {
            grid.set(Coord<2>(i, 0), MyDummyCell(47, 11, i + 'a'));
        }

        std::vector<double> targetZ(20);
        grid.saveMember(&targetZ[0], selectorZ, region);

        for (int i = 0; i < 20; ++i) {
            TS_ASSERT_EQUALS(i + 'a' + 20, targetZ[i]);
        }

        // test copyStreakIn:
        grid.loadMember(&targetZ[0], selectorZ, region);

        for (int i = 0; i < 20; ++i) {
            char expected = i;
            expected += 30 + 'a';
            TS_ASSERT_EQUALS(grid.get(Coord<2>(i, 0)).z, expected);
        }
    }

    void testArrayMemberWithDefaultFilter1()
    {
        Selector<MyOtherDummyCell> selectorY(&MyOtherDummyCell::y, "varY");

        Coord<2> dim(20, 10);

        // test copyMemberOut
        Region<2> region;
        region << Streak<2>(Coord<2>(5, 0), 15)
               << Streak<2>(Coord<2>(7, 6), 19)
               << Streak<2>(Coord<2>(0, 9), 20);
        TS_ASSERT_EQUALS(std::size_t(42), region.size());

        std::vector<double> targetY(3 * region.size(), -1);
        Grid<MyOtherDummyCell> grid(dim);

        for (int y = 0; y < dim.y(); ++y) {
            for (int x = 0; x < dim.x(); ++x) {
                grid[Coord<2>(x, y)].y[0] = x;
                grid[Coord<2>(x, y)].y[1] = y;
                grid[Coord<2>(x, y)].y[2] = 47.11;
            }
        }

        grid.saveMember(&targetY[0], selectorY, region);

        Region<2>::Iterator i = region.begin();
        for (std::size_t j = 0; j < targetY.size(); j += 3) {
            TS_ASSERT_EQUALS(targetY[j + 0], i->x());
            TS_ASSERT_EQUALS(targetY[j + 1], i->y());
            TS_ASSERT_EQUALS(targetY[j + 2], 47.11);
            ++i;
        }

        // test copyMemberIn
        region.clear();
        region << Streak<2>(Coord<2>(0, 4), 10)
               << Streak<2>(Coord<2>(5, 9), 20);
        TS_ASSERT_EQUALS(25, region.size());

        std::vector<double> sourceY(3 * region.size(), -1);
        for (std::size_t i = 0; i < region.size(); ++i) {
            sourceY[i * 3 + 0] = i;
            sourceY[i * 3 + 1] = 12.34;
            sourceY[i * 3 + 2] = i * 3.0 + 5.0;
        }

        grid.loadMember(&sourceY[0], selectorY, region);

        i = region.begin();
        for (std::size_t j = 0; j < region.size(); ++j) {
            TS_ASSERT_EQUALS(grid[*i].y[0], j);
            TS_ASSERT_EQUALS(grid[*i].y[1], 12.34);
            TS_ASSERT_EQUALS(grid[*i].y[2], j * 3.0 + 5.0);
            ++i;
        }
    }

    void testArrayMemberWithDefaultFilter2()
    {
        Selector<MyOtherDummyCell> selectorY(&MyOtherDummyCell::y, "varY");

        Coord<2> dim(20, 10);

        // test copyStreakOut
        Region<2> region;
        region << Streak<2>(Coord<2>(5, 0), 15)
               << Streak<2>(Coord<2>(7, 6), 19)
               << Streak<2>(Coord<2>(0, 9), 20);
        TS_ASSERT_EQUALS(42, region.size());

        std::vector<double> targetY(3 * region.size(), -1);
        SoAGrid<MyOtherDummyCell> grid(CoordBox<2>(Coord<2>(), dim));

        for (int y = 0; y < dim.y(); ++y) {
            for (int x = 0; x < dim.x(); ++x) {
                MyOtherDummyCell cell;
                cell.y[0] = y;
                cell.y[1] = x;
                cell.y[2] = 42.23;
                grid.set(Coord<2>(x, y), cell);
            }
        }

        grid.saveMember(&targetY[0], selectorY, region);

        Region<2>::Iterator i = region.begin();
        for (std::size_t j = 0; j < targetY.size(); j += 3) {
            TS_ASSERT_EQUALS(targetY[j + 0], i->y());
            TS_ASSERT_EQUALS(targetY[j + 1], i->x());
            TS_ASSERT_EQUALS(targetY[j + 2], 42.23);
            ++i;
        }

        // test copyStreakIn
        region.clear();
        region << Streak<2>(Coord<2>(0, 4), 10)
               << Streak<2>(Coord<2>(5, 9), 20);
        TS_ASSERT_EQUALS(25, region.size());

        std::vector<double> sourceY(3 * region.size(), -1);
        for (std::size_t i = 0; i < region.size(); ++i) {
            sourceY[i * 3 + 0] = i + 1000;
            sourceY[i * 3 + 1] = 56.78;
            sourceY[i * 3 + 2] = i * 7.0 + 555.0;
        }

        grid.loadMember(&sourceY[0], selectorY, region);

        i = region.begin();
        for (std::size_t j = 0; j < region.size(); ++j) {
            MyOtherDummyCell cell = grid.get(*i);
            TS_ASSERT_EQUALS(cell.y[0], j + 1000);
            TS_ASSERT_EQUALS(cell.y[1], 56.78);
            TS_ASSERT_EQUALS(cell.y[2], j * 7.0 + 555.0);
            ++i;
        }
    }
};

}
