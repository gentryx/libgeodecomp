#include <libgeodecomp/io/selector.h>
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

        TS_ASSERT_EQUALS(sizeof(int),    selectorX.sizeOf());
        TS_ASSERT_EQUALS(sizeof(double), selectorY.sizeOf());
        TS_ASSERT_EQUALS(sizeof(char),   selectorZ.sizeOf());

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
};

}
