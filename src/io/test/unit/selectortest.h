#include <libgeodecomp/io/selector.h>
#include <libgeodecomp/misc/color.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/storage/soagrid.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class MyDummyCell
{
public:
    class API :
        public APITraits::HasSoA
    {};

    MyDummyCell(const int x = 0, const double y = 0, const char z = 0) :
        x(x),
        y(y),
        z(z)
    {}

    int x;
    double y;
    char z;
};

}

LIBFLATARRAY_REGISTER_SOA(LibGeoDecomp::MyDummyCell, ((int)(x))((double)(y))((char)(z)) )

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

        TS_ASSERT_EQUALS(sizeof(int),    selectorX.sizeOfMember());
        TS_ASSERT_EQUALS(sizeof(double), selectorY.sizeOfMember());
        TS_ASSERT_EQUALS(sizeof(char),   selectorZ.sizeOfMember());

        TS_ASSERT_EQUALS(sizeof(int),    selectorX.sizeOfExternal());
        TS_ASSERT_EQUALS(sizeof(double), selectorY.sizeOfExternal());
        TS_ASSERT_EQUALS(sizeof(char),   selectorZ.sizeOfExternal());

        TS_ASSERT_EQUALS( 0, selectorX.offset());
        TS_ASSERT_EQUALS( 4, selectorY.offset());
        TS_ASSERT_EQUALS(12, selectorZ.offset());
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

        std::vector<int>    targetX(20, -1);
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
        std::vector<int>    targetX;
        std::vector<double> targetY;
        std::vector<char>   targetZ;

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

    class MyDummyFilter : public Selector<MyDummyCell>::Filter<double, Color>
    {
    public:
        void copyStreakInImpl(const Color *first, const Color *last, double *target)
        {
            for (const Color *i = first; i != last; ++i, ++target) {
                *target = i->red() * 2 + 10;
            }
        }

        void copyStreakOutImpl(const double *first, const double *last, Color *target)
        {
            for (const double *i = first; i != last; ++i, ++target) {
                *target = Color(*i, 47, 11);
            }
        }

        void copyMemberInImpl(
            const Color *source, MyDummyCell *target, int num, double MyDummyCell:: *memberPointer)
        {
            for (int i = 0; i < num; ++i) {
                target[i].*memberPointer = source[i].red() * 2 + 10;
            }
        }

        void copyMemberOutImpl(
            const MyDummyCell *source, Color *target, int num, double MyDummyCell:: *memberPointer)
        {
            for (int i = 0; i < num; ++i) {
                target[i] = Color(source[i].*memberPointer, 47, 11);
            }
        }
    };

    class MySimpleFilter : public Selector<MyDummyCell>::SimpleFilter<char, double>
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
        class FancyFilter : public Selector<MyDummyCell>::SimpleFilter<char, double>
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

        Selector<MyDummyCell>::FilterBase *filter1 = new FancyFilter();
        boost::shared_ptr<Selector<MyDummyCell>::FilterBase> filter2(filter1);
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
        boost::shared_ptr<Selector<MyDummyCell>::FilterBase> filter(
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
        boost::shared_ptr<Selector<MyDummyCell>::FilterBase> filter(
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
        boost::shared_ptr<Selector<MyDummyCell>::FilterBase> filter(
            new MyDummyFilter());
        Selector<MyDummyCell> selectorY(&MyDummyCell::y, "varY", filter);

        selectorY.checkTypeID<Color>();

        std::vector<double> vec;
        for (int i = 0; i < 20; ++i) {
            vec << i + 50;
        }

        std::vector<Color> targetY(20);
        selectorY.copyStreakOut((char*)&vec[0], (char*)(&vec[0] + 20), (char*)&targetY[0]);

        for (int i = 0; i < 20; ++i) {
            TS_ASSERT_EQUALS(Color(i + 50, 47, 11), targetY[i]);
        }

        // test copyStreakIn:
        selectorY.copyStreakIn((char*)&targetY[0], (char*)(&targetY[0] + 20), (char*)&vec[0]);

        for (int i = 0; i < 20; ++i) {
            TS_ASSERT_EQUALS(vec[i], (50 + i) * 2 + 10);
        }
    }

    void testFilterSoA2()
    {
        // test copyStreakOut:
        boost::shared_ptr<Selector<MyDummyCell>::FilterBase> filter(
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
        boost::shared_ptr<Selector<MyDummyCell>::FilterBase> filter(
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
};

}
