#ifndef _libgeodecomp_misc_updatefunctortestbase_h_
#define _libgeodecomp_misc_updatefunctortestbase_h_

#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/misc/testhelper.h>

namespace LibGeoDecomp {

template<class STENCIL>
class UpdateFunctorTestBase
{
public:
    typedef STENCIL Stencil;
    const static int DIM = Stencil::DIM;
    typedef TestCell<DIM, Stencil> TestCellType;
    typedef Grid<TestCellType, typename TestCellType::Topology> GridType;

    virtual ~UpdateFunctorTestBase()
    {}

    void checkStencil(int steps)
    {

        Coord<DIM> dim;
        for (int d = 0; d < DIM; ++d) {
            dim[d] = (d + 1) * 5;
        }

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

                callFunctor(streak, gridOld, &gridNew, s);
            }
                                     
            int cycle = init.startStep() * TestCellType::nanoSteps() + s;
            TS_ASSERT_TEST_GRID2(GridType, gridOld, cycle, typename);
            cycle += 1;
            TS_ASSERT_TEST_GRID2(GridType, gridNew, cycle, typename);

            std::swap(gridOld, gridNew);
        }
    }

    virtual void callFunctor(
        const Streak<DIM>& streak,
        const GridType& gridOld,
        GridType *gridNew,
        unsigned nanoStep) = 0;
};

}

#endif
