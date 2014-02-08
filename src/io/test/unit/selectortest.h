#include <libgeodecomp/io/selector.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/storage/soagrid.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class MyDummyCell
{
public:
    MyDummyCell(const int x = 0, const double y = 0) :
        x(x),
        y(y)
    {}

    int x;
    double y;
};

}

namespace LibGeoDecomp {

class SelectorTest : public CxxTest::TestSuite
{
public:

    void testAoS()
    {
        Selector<MyDummyCell> selectorX(&MyDummyCell::x, "varX");
        Selector<MyDummyCell> selectorY(&MyDummyCell::y, "varY");

        std::vector<MyDummyCell> vec;
        for (int i = 0; i < 20; ++i) {
            vec << MyDummyCell(i, 47.11 + i);
        }

        std::vector<int> targetX(20, -1);
        std::vector<double> targetY(20, -1);
        selectorX(&vec[0], &targetX[0], 20);
        selectorY(&vec[0], &targetY[0], 20);

        for (int i = 0; i < 20; ++i) {
            TS_ASSERT_EQUALS(targetX[i], i);
            TS_ASSERT_EQUALS(targetY[i], 47.11 + i);
        }

        TS_ASSERT_EQUALS("varX", selectorX.name());
        TS_ASSERT_EQUALS("varY", selectorY.name());

        std::cout << "wooo!\n";
    }

};

}
