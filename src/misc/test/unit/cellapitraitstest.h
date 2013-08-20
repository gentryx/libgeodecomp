#include <libgeodecomp/misc/cellapitraits.h>
#include <libgeodecomp/misc/updatefunctor.h>

using namespace LibGeoDecomp;

std::ostringstream myTestEvents;

class MySimpleDummyCell
{
public:
    typedef Stencils::Moore<2, 1> Stencil;

    class API : public CellAPITraits::Base
    {};

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, unsigned nanoStep)
    {
        myTestEvents << "MySimpleDummyCell::update()\n";
    }
};

class MyFancyDummyCell
{
public:
    typedef Stencils::Moore<3, 1> Stencil;

    class API :
        public CellAPITraits::Base,
        public CellAPITraitsFixme::HasTorusTopology<3>
    {};

    static unsigned nanoSteps()
    {
        return 3;
    }

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, unsigned nanoStep)
    {
        myTestEvents << "MyFancyDummyCell::update()\n";
    }
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
        typedef CellAPITraitsFixme::SelectTopology<MySimpleDummyCell>::Value Topology;
        Grid<MySimpleDummyCell, Topology> gridOld(gridDim);
        Grid<MySimpleDummyCell, Topology> gridNew(gridDim);
        Streak<2> streak(Coord<2>(0, 0), 5);
        Coord<2> targetOffset(0, 0);
        UpdateFunctor<MySimpleDummyCell>()(streak, targetOffset, gridOld, &gridNew, 0);

        std::string expected;
        for (int i = 0; i < streak.length(); ++i) {
            expected += "MySimpleDummyCell::update()\n";
        }
        TS_ASSERT_EQUALS(myTestEvents.str(), expected);
    }

    void testBasicCallback2()
    {
        Coord<3> gridDim(10, 5, 10);
        typedef CellAPITraitsFixme::SelectTopology<MyFancyDummyCell>::Value Topology;
        Grid<MyFancyDummyCell, Topology> gridOld(gridDim);
        Grid<MyFancyDummyCell, Topology> gridNew(gridDim);
        Streak<3> streak(Coord<3>(0, 0, 0), 4);
        Coord<3> targetOffset(0, 0, 0);
        UpdateFunctor<MyFancyDummyCell>()(streak, targetOffset, gridOld, &gridNew, 0);

        std::string expected;
        for (int i = 0; i < streak.length(); ++i) {
            expected += "MyFancyDummyCell::update()\n";
        }
        TS_ASSERT_EQUALS(myTestEvents.str(), expected);
    }
};

}
