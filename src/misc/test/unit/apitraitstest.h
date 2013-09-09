#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/updatefunctor.h>
#include <libgeodecomp/misc/stringops.h>

using namespace LibGeoDecomp;

std::ostringstream myTestEvents;

class MySimpleDummyCell
{
public:
    MySimpleDummyCell(int val = 0) :
        val(val)
    {};

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, unsigned nanoStep)
    {
        myTestEvents << "MySimpleDummyCell::update(" << hood[FixedCoord<-1, 0, 0>()].val << ")\n";
    }

    int val;
};

class MyFancyDummyCell
{
public:
    class API :
        public APITraits::HasTorusTopology<3>,
        public APITraits::HasStencil<Stencils::Moore<3, 1> >,
        public APITraits::HasNanoSteps<3>
    {};

    MyFancyDummyCell(int val = 0) :
        val(val)
    {};

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, unsigned nanoStep)
    {
        myTestEvents << "MyFancyDummyCell::update(" << hood[FixedCoord<-1, 0, 0>()].val << ")\n";
    }

    int val;
};

namespace LibGeoDecomp {

class CellAPITraitsTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
        myTestEvents.str("");
    }

    void testBasicCallback1()
    {
        Coord<2> gridDim(10, 5);
        typedef APITraits::SelectTopology<MySimpleDummyCell>::Value Topology;
        Grid<MySimpleDummyCell, Topology> gridOld(gridDim, 666, 31);
        Grid<MySimpleDummyCell, Topology> gridNew(gridDim);

        Streak<2> streak(Coord<2>(0, 0), 5);
        Region<2> region;
        region << streak;
        UpdateFunctor<MySimpleDummyCell>()(region, Coord<2>(), Coord<2>(), gridOld, &gridNew, 0);

        std::string expected;
        for (int i = 0; i < streak.length(); ++i) {
            int num = 666;
            if (i == 0) {
                num = 31;
            }
            expected += "MySimpleDummyCell::update(" + StringOps::itoa(num) + ")\n";
        }
        TS_ASSERT_EQUALS(myTestEvents.str(), expected);
    }

    void testBasicCallback2()
    {
        Coord<3> gridDim(10, 5, 10);
        typedef APITraits::SelectTopology<MyFancyDummyCell>::Value Topology;
        Grid<MyFancyDummyCell, Topology> gridOld(gridDim, 666, 31);
        Grid<MyFancyDummyCell, Topology> gridNew(gridDim);

        Streak<3> streak(Coord<3>(0, 0, 0), 4);
        Region<3> region;
        region << streak;
        UpdateFunctor<MyFancyDummyCell>()(region, Coord<3>(), Coord<3>(), gridOld, &gridNew, 0);

        std::string expected;
        for (int i = 0; i < streak.length(); ++i) {
            expected += "MyFancyDummyCell::update(666)\n";
        }
        TS_ASSERT_EQUALS(myTestEvents.str(), expected);
    }
};

}
