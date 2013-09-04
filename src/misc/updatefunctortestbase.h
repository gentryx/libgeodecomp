#ifndef LIBGEODECOMP_MISC_UPDATEFUNCTORTESTBASE_H
#define LIBGEODECOMP_MISC_UPDATEFUNCTORTESTBASE_H

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
    typedef Grid<TestCellType, typename APITraits::SelectTopology<TestCellType>::Value> GridType;

    static const unsigned NANO_STEPS = APITraits::SelectNanoSteps<TestCellType>::VALUE;

    virtual ~UpdateFunctorTestBase()
    {}

    void testSimple(int steps)
    {
        Coord<DIM> dim;
        for (int d = 0; d < DIM; ++d) {
            dim[d] = d * 5 + 10;
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

            int cycle = init.startStep() * NANO_STEPS + s;
            TS_ASSERT_TEST_GRID2(GridType, gridOld, cycle, typename);
            cycle += 1;
            TS_ASSERT_TEST_GRID2(GridType, gridNew, cycle, typename);

            std::swap(gridOld, gridNew);
        }
    }

    void testSplittedTraversal(int steps)
    {
        Coord<DIM> dim;
        for (int d = 0; d < DIM; ++d) {
            dim[d] = d * 5 + 10;
        }

        TestInitializer<TestCellType> init(dim);
        GridType gridOld(dim);
        init.grid(&gridOld);
        GridType gridNew = gridOld;
        int halfWidth = dim.x() / 2;

        CoordBox<DIM> lineStarts = gridOld.boundingBox();
        lineStarts.dimensions.x() = 1;

        for (int s = 0; s < steps; ++s) {
            for (typename CoordBox<DIM>::Iterator i = lineStarts.begin();
                 i != lineStarts.end();
                 ++i) {
                Coord<DIM> origin1 = *i;
                Coord<DIM> origin2 = *i;
                origin2.x() = halfWidth;
                Streak<DIM> s1(origin1, halfWidth);
                Streak<DIM> s2(origin2, dim.x());

                callFunctor(s2, gridOld, &gridNew, s);
                callFunctor(s1, gridOld, &gridNew, s);
            }

            int cycle = init.startStep() * NANO_STEPS + s;
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
