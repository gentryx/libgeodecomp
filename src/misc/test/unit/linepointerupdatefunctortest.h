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

    template<class STENCIL>
    void checkStencil(int steps)
    {
        typedef STENCIL Stencil;
        const int DIM = Stencil::DIM;
        typedef TestCell<DIM, Stencil> TestCellType;
        typedef Grid<TestCellType, typename TestCellType::Topology> GridType;

        Coord<DIM> dim;
        for (int d = 0; d < DIM; ++d) {
            dim[d] = (d + 1) * 5;
        }

        CoordBox<DIM> gridBox(Coord<DIM>(), dim);
        TestInitializer<TestCellType> init(dim);
        GridType gridOld(dim);
        init.grid(&gridOld);
        GridType gridNew = gridOld;
        
        CoordBox<DIM> lineStarts = gridOld.boundingBox();
        lineStarts.dimensions.x() = 1;

        for (int s = 0; s < steps; ++s) {
            for (typename CoordBox<DIM>::Iterator i = lineStarts.begin();
                 i != lineStarts.end();
                 ++i) {
                Streak<DIM> streak(*i, dim.x());
                TestCellType *pointers[Stencil::VOLUME];
                LinePointerAssembly<Stencil>()(pointers, streak, gridOld);
                LinePointerUpdateFunctor<TestCellType>()(
                    streak, gridBox, pointers, &gridNew[streak.origin], s);
            }
                                     
            int cycle = init.startStep() * TestCellType::nanoSteps() + s;
            TS_ASSERT_TEST_GRID(GridType, gridOld, cycle);
            cycle += 1;
            TS_ASSERT_TEST_GRID(GridType, gridNew, cycle);

            std::swap(gridOld, gridNew);
        }
    }

    void testMoore2D()
    {
        checkStencil<Stencils::Moore<2, 1> >(3);
    }

    void testMoore3D()
    {
        checkStencil<Stencils::Moore<3, 1> >(3);
    }

    void testVonNeumann2D()
    {
        checkStencil<Stencils::VonNeumann<2, 1> >(3);
    }

    void testVonNeumann3D()
    {
        checkStencil<Stencils::VonNeumann<3, 1> >(3);
    }
};

}
