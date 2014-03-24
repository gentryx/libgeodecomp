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

    class MyDummyFilter : public Selector<MyDummyCell>::Filter<Color>
    {
    public:

        void copyStreakIn(const char *first, const char *last, char *target)
        {
            // fixme: users shouldn't have to do their own type casting!
            const Color *actualFirst = reinterpret_cast<const Color*>(first);
            const Color *actualLast = reinterpret_cast<const Color*>(last);
            double *actualTarget = reinterpret_cast<double*>(target);

            for (const Color *i = actualFirst; i != actualLast; ++i, ++actualTarget) {
                *actualTarget = i->red() * 2 + 10;
            }
        }

        void copyStreakOut(const char *first, const char *last, char *target)
        {
            // fixme: users shouldn't have to do their own type casting!
            const double *actualFirst = reinterpret_cast<const double*>(first);
            const double *actualLast = reinterpret_cast<const double*>(last);
            Color *actualTarget = reinterpret_cast<Color*>(target);

            for (const double *i = actualFirst; i != actualLast; ++i, ++actualTarget) {
                *actualTarget = Color(*i, 47, 11);
            }
        }

        void copyMemberIn(
            const char *source, MyDummyCell *target, int num, char MyDummyCell:: *memberPointer)
        {
            // fixme: users shouldn't have to do their own type casting!
            double MyDummyCell:: *actualMember = reinterpret_cast<double MyDummyCell:: *>(memberPointer);
            const Color *cursor = reinterpret_cast<const Color*>(source);

            for (int i = 0; i < num; ++i) {
                target[i].*actualMember = cursor[i].red() * 2 + 10;
            }
        }

        void copyMemberOut(
            const MyDummyCell *source, char *target, int num, char MyDummyCell:: *memberPointer)
        {
            // fixme: users shouldn't have to do their own type casting!
            double MyDummyCell:: *actualMember = reinterpret_cast<double MyDummyCell:: *>(memberPointer);
            Color *cursor = reinterpret_cast<Color*>(target);

            for (int i = 0; i < num; ++i) {
                cursor[i] = Color(source[i].*actualMember, 47, 11);
            }
        }
    };

    void testFilterAoS()
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
};

}
