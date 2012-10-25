#include <sstream>
#include <vector>
#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/misc/testhelper.h>
#include <libgeodecomp/misc/updatefunctor.h>
#include <libgeodecomp/misc/updatefunctortestbase.h>

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

std::stringstream log;

class BasicCell
{
public:
    typedef Stencils::Moore<2, 1> Stencil;
    typedef Topologies::Torus<2>::Topology Topology;

    class API : public APIs::Base
    {};

    static int nanoSteps()
    {
        return 1;
    }

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, int nanoStep)
    {
        log << "BasicCell::update(nanoStep = " << nanoStep << ")\n";
    }

};

class LineUpdateCell
{
public:
    typedef Stencils::Moore<2, 1> Stencil;
    typedef Topologies::Torus<2>::Topology Topology;

    class API : public APIs::Line
    {};

    static int nanoSteps()
    {
        return 1;
    }

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, int nanoStep)
    {
        log << "LineUpdateCell::update(nanoStep = " << nanoStep << ")\n";
    }

    // won't ever be called as all current update functors support
    // updateLine only with fixed neighborhoods
    template<typename NEIGHBORHOOD>
    static void updateLine(LineUpdateCell *target, long *x, long endX, const NEIGHBORHOOD& hood, int nanoStep)
    {
        log << "LineUpdateCell::updateLine(x = " << *x << ", endX = " << endX << ", nanoStep = " << nanoStep << ")\n";
    
    }
};

class FixedCell
{
public:
    typedef Stencils::Moore<2, 1> Stencil;
    typedef Topologies::Torus<2>::Topology Topology;

    class API : public APIs::Fixed
    {};

    static int nanoSteps()
    {
        return 1;
    }

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, int nanoStep)
    {
        log << "FixedCell::update(nanoStep = " << nanoStep << ")\n";
    }

};

class FixedLineUpdateCell
{
public:
    typedef Stencils::Moore<2, 1> Stencil;
    typedef Topologies::Torus<2>::Topology Topology;

    class API : public APIs::Fixed, public APIs::Line
    {};

    static int nanoSteps()
    {
        return 1;
    }

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, int nanoStep)
    {
        log << "FixedLineUpdateCell::update(nanoStep = " << nanoStep << ")\n";
    }

    template<typename NEIGHBORHOOD>
    static void updateLine(FixedLineUpdateCell *target, long *x, long endX, const NEIGHBORHOOD& hood, int nanoStep)
    {
        log << "FixedLineUpdateCell::updateLine(x = " << *x << ", endX = " << endX << ", nanoStep = " << nanoStep << ")\n";
    
    }
};

class UpdateFunctorTest : public CxxTest::TestSuite 
{
public:
    template<class STENCIL>
    class UpdateFunctorTestHelper : public UpdateFunctorTestBase<STENCIL>
    {
    public:
        using UpdateFunctorTestBase<STENCIL>::DIM;
        typedef typename UpdateFunctorTestBase<STENCIL>::TestCellType TestCellType;
        typedef typename UpdateFunctorTestBase<STENCIL>::GridType GridType;
        typedef STENCIL Stencil;

        virtual void callFunctor(
            const Streak<DIM>& streak,
            const GridType& gridOld,
            GridType *gridNew,
            unsigned nanoStep) 
        {
            UpdateFunctor<TestCellType>()(
                streak, gridOld, gridNew, nanoStep);
        }
    };

    void testSelector()
    {
        checkSelector<BasicCell>(
            "BasicCell::update(nanoStep = 0)\n", 8);
        checkSelector<LineUpdateCell>(
            "LineUpdateCell::update(nanoStep = 0)\n", 8);
        checkSelector<FixedCell>(
            "FixedCell::update(nanoStep = 0)\n", 8);
        checkSelector<FixedLineUpdateCell>(
            "FixedLineUpdateCell::update(nanoStep = 0)\nFixedLineUpdateCell::updateLine(x = 2, endX = 9, nanoStep = 0)\nFixedLineUpdateCell::update(nanoStep = 0)\n", 1);
    }

    void testMoore2D()
    {
        UpdateFunctorTestHelper<Stencils::Moore<2, 1> >().checkStencil(3);
    }

    void testMoore3D()
    {
        UpdateFunctorTestHelper<Stencils::Moore<3, 1> >().checkStencil(3);
    }

    void testVonNeumann2D()
    {
        UpdateFunctorTestHelper<Stencils::VonNeumann<2, 1> >().checkStencil(3);
    }

    void testVonNeumann3D()
    {
        UpdateFunctorTestHelper<Stencils::VonNeumann<3, 1> >().checkStencil(3);
    }

private:
    template<typename CELL>
    void checkSelector(const std::string& line, int repeats)
    {
        Streak<2> streak(Coord<2>(2, 1), 10);
        Coord<2> dim(20, 10);
        int nanoStep = 0;

        Grid<CELL> gridOld(dim);
        Grid<CELL> gridNew(dim);

        UpdateFunctor<CELL>()(streak, gridOld, &gridNew, nanoStep);

        std::vector<char> message(1024 * 16, 0);
        log.read(&message[0], 1024 * 16);
        std::string expected = "";
        for (int i = 0; i < repeats; ++i) {
            expected += line;
        }
        TS_ASSERT_EQUALS(expected, std::string(&message[0]));
        log.clear();
    }
    
};

}
