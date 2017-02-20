#ifndef LIBGEODECOMP_STORAGE_UPDATEFUNCTORTESTBASE_H
#define LIBGEODECOMP_STORAGE_UPDATEFUNCTORTESTBASE_H

#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/storage/grid.h>

namespace LibGeoDecomp {

/**
 * This class is used to build test suites for our various UpdateFunctor types.
 */
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
        using std::swap;
        Coord<DIM> dim;
        for (int d = 0; d < DIM; ++d) {
            dim[d] = d * 5 + 10;
        }

        TestInitializer<TestCellType> init(dim);
        GridType gridOld(dim);
        init.grid(&gridOld);
        GridType gridNew = gridOld;

        Region<DIM> region;
        region << gridOld.boundingBox();


        for (int s = 0; s < steps; ++s) {
            callFunctor(region, gridOld, &gridNew, s);

            int cycle = init.startStep() * NANO_STEPS + s;
            TS_ASSERT_TEST_GRID2(GridType, gridOld, cycle, typename);
            cycle += 1;
            TS_ASSERT_TEST_GRID2(GridType, gridNew, cycle, typename);

            swap(gridOld, gridNew);
        }
    }

    void testSplittedTraversal(int steps)
    {
        using std::swap;
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

        for (int s = 0; s < steps; ++s) {
            for (typename CoordBox<DIM>::StreakIterator i = lineStarts.beginStreak();
                 i != lineStarts.endStreak();
                 ++i) {
                Streak<DIM> streak = *i;

                Coord<DIM> origin1 = streak.origin;
                Coord<DIM> origin2 = streak.origin;
                origin2.x() = halfWidth;
                Streak<DIM> s1(origin1, halfWidth);
                Streak<DIM> s2(origin2, dim.x());

                Region<DIM> r1;
                Region<DIM> r2;
                r1 << s1;
                r2 << s2;

                callFunctor(r2, gridOld, &gridNew, s);
                callFunctor(r1, gridOld, &gridNew, s);
            }

            int cycle = init.startStep() * NANO_STEPS + s;
            TS_ASSERT_TEST_GRID2(GridType, gridOld, cycle, typename);
            cycle += 1;
            TS_ASSERT_TEST_GRID2(GridType, gridNew, cycle, typename);

            swap(gridOld, gridNew);
        }
    }

    virtual void callFunctor(
        const Region<DIM>& region,
        const GridType& gridOld,
        GridType *gridNew,
        unsigned nanoStep) = 0;
};

}

#endif
