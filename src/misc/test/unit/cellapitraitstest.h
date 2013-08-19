#include <libgeodecomp/misc/cellapitraits.h>
#include <libgeodecomp/misc/updatefunctor.h>

using namespace LibGeoDecomp;

std::ostringstream myCellAPITestEvents;

class MyDummyCell
{
public:
    typedef Topologies::Cube<2>::Topology Topology;
    typedef Stencils::Moore<2, 1> Stencil;

    class API : public CellAPITraits::Base
    {};

    static unsigned nanoSteps()
    {
        return 1;
    }

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, unsigned nanoStep)
    {
        myCellAPITestEvents << "MyDummyCell::update()\n";
    }
};

namespace LibGeoDecomp {

class CellAPITraitsTest : public CxxTest::TestSuite
{
public:
    void testBasic()
    {
        std::cout << "foo-------------------------------------------------------------------------------\n";
        Coord<2> gridDim(10, 5);
        Grid<MyDummyCell> gridOld(gridDim);
        Grid<MyDummyCell> gridNew(gridDim);
        Streak<2> streak(Coord<2>(0, 0), 4);
        Coord<2> targetOffset(0, 0);
        UpdateFunctor<MyDummyCell>()(streak, targetOffset, gridOld, &gridNew, 0);

        std::cout << myCellAPITestEvents.str() << "\n";
    }
};

}
