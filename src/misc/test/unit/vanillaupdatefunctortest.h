#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/updatefunctortestbase.h>
#include <libgeodecomp/misc/vanillaupdatefunctor.h>

using namespace LibGeoDecomp; 

namespace LibGeoDecomp {

class VanillaUpdateFunctorTest : public CxxTest::TestSuite
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
            VanillaUpdateFunctor<TestCellType>()(
                streak, gridOld, gridNew, nanoStep);
        }
    };

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
};

}
