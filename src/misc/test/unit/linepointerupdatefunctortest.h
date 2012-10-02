#include <cxxtest/TestSuite.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/misc/linepointerassembly.h>
#include <libgeodecomp/misc/linepointerupdatefunctor.h>

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class LinePointerUpdateFunctorTest : public CxxTest::TestSuite 
{
public:

    void testMoore2D()
    {
        typedef Stencils::Moore<2, 1> Stencil;
        typedef Grid<TestCell<2> > GridType;

        Coord<2> dim(10, 5);
        CoordBox<2> gridBox(Coord<2>(), dim);
        TestInitializer<TestCell<2> > init(dim);
        GridType gridOld(dim);
        init.grid(&gridOld);
        GridType gridNew = gridOld;
        
        for (int y = 0; y < dim.y(); ++y) {
            Streak<2> streak(Coord<2>(0, y), dim.x());
            TestCell<2> *pointers[Stencil::VOLUME];
            LinePointerAssembly<Stencil>()(pointers, streak, gridOld);
            LinePointerUpdateFunctor<TestCell<2> >()(
                        streak, gridBox, pointers, &gridNew[streak.origin]);
        }
                                     
        int cycle = init.startStep() * TestCell<2>::nanoSteps();
        TS_ASSERT_TEST_GRID(GridType, gridOld, cycle);
        cycle += 1;
        TS_ASSERT_TEST_GRID(GridType, gridNew, cycle);
    }

    void testVonNeumann2D()
    {
        typedef Stencils::VonNeumann<2, 1> Stencil;
        typedef TestCell<2, Stencil> TestCellType;
        typedef Grid<TestCellType> GridType;

        Coord<2> dim(10, 5);
        CoordBox<2> gridBox(Coord<2>(), dim);
        TestInitializer<TestCellType> init(dim);
        GridType gridOld(dim);
        init.grid(&gridOld);
        GridType gridNew = gridOld;
        
        for (int y = 0; y < dim.y(); ++y) {
            Streak<2> streak(Coord<2>(0, y), dim.x());
            TestCellType *pointers[Stencil::VOLUME];
            LinePointerAssembly<Stencil>()(pointers, streak, gridOld);
            LinePointerUpdateFunctor<TestCellType>()(
                streak, gridBox, pointers, &gridNew[streak.origin]);
        }
                                     
        int cycle = init.startStep() * TestCellType::nanoSteps();
        TS_ASSERT_TEST_GRID(GridType, gridOld, cycle);
        cycle += 1;
        TS_ASSERT_TEST_GRID(GridType, gridNew, cycle);
    }
};

}
