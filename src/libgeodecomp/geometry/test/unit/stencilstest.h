#include <cxxtest/TestSuite.h>
#include <libgeodecomp/geometry/stencils.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class StencilsTest : public CxxTest::TestSuite
{
public:
    void testPower()
    {
        TS_ASSERT_EQUALS(9, (Stencils::Power<3, 2>::VALUE));
        TS_ASSERT_EQUALS(8, (Stencils::Power<2, 3>::VALUE));
        TS_ASSERT_EQUALS(1, (Stencils::Power<7, 0>::VALUE));
    }

    void testSum1()
    {
        TS_ASSERT_EQUALS( 1, (Stencils::Sum1<Stencils::Moore, 0, 1>::VALUE));
        TS_ASSERT_EQUALS( 4, (Stencils::Sum1<Stencils::Moore, 1, 1>::VALUE));
        TS_ASSERT_EQUALS(13, (Stencils::Sum1<Stencils::Moore, 2, 1>::VALUE));
    }

    void testSum2()
    {
        TS_ASSERT_EQUALS( 1, (Stencils::Sum2<Stencils::Moore, 2, 0>::VALUE));
        TS_ASSERT_EQUALS(10, (Stencils::Sum2<Stencils::Moore, 2, 1>::VALUE));
        TS_ASSERT_EQUALS(35, (Stencils::Sum2<Stencils::Moore, 2, 2>::VALUE));
    }

    void testOffsetHelper()
    {
        TS_ASSERT_EQUALS( 0, (Stencils::OffsetHelper<Stencils::Moore<1, 1>, -1,  0, 0>::VALUE));

        TS_ASSERT_EQUALS( 0, (Stencils::OffsetHelper<Stencils::Moore<1, 2>, -2,  0, 0>::VALUE));
        TS_ASSERT_EQUALS( 4, (Stencils::OffsetHelper<Stencils::Moore<1, 2>,  2,  0, 0>::VALUE));

        TS_ASSERT_EQUALS( 4, (Stencils::OffsetHelper<Stencils::Moore<2, 1>,  0,  0, 0>::VALUE));
        TS_ASSERT_EQUALS( 3, (Stencils::OffsetHelper<Stencils::Moore<2, 1>, -1,  0, 0>::VALUE));
        TS_ASSERT_EQUALS( 0, (Stencils::OffsetHelper<Stencils::Moore<2, 1>, -1, -1, 0>::VALUE));
        TS_ASSERT_EQUALS( 3, (Stencils::OffsetHelper<Stencils::Moore<2, 1>, -1,  0, 0>::VALUE));
        TS_ASSERT_EQUALS( 7, (Stencils::OffsetHelper<Stencils::Moore<2, 1>,  0,  1, 0>::VALUE));
        TS_ASSERT_EQUALS( 8, (Stencils::OffsetHelper<Stencils::Moore<2, 1>,  1,  1, 0>::VALUE));

        TS_ASSERT_EQUALS(12, (Stencils::OffsetHelper<Stencils::Moore<2, 2>,  0,  0, 0>::VALUE));
        TS_ASSERT_EQUALS(11, (Stencils::OffsetHelper<Stencils::Moore<2, 2>, -1,  0, 0>::VALUE));
        TS_ASSERT_EQUALS( 7, (Stencils::OffsetHelper<Stencils::Moore<2, 2>,  0, -1, 0>::VALUE));
        TS_ASSERT_EQUALS( 0, (Stencils::OffsetHelper<Stencils::Moore<2, 2>, -2, -2, 0>::VALUE));
        TS_ASSERT_EQUALS(17, (Stencils::OffsetHelper<Stencils::Moore<2, 2>,  0,  1, 0>::VALUE));

        TS_ASSERT_EQUALS( 0, (Stencils::OffsetHelper<Stencils::Moore<3, 1>, -1, -1, -1>::VALUE));
        TS_ASSERT_EQUALS(10, (Stencils::OffsetHelper<Stencils::Moore<3, 1>,  0, -1,  0>::VALUE));
        TS_ASSERT_EQUALS(13, (Stencils::OffsetHelper<Stencils::Moore<3, 1>,  0,  0,  0>::VALUE));
        TS_ASSERT_EQUALS(25, (Stencils::OffsetHelper<Stencils::Moore<3, 1>,  0,  1,  1>::VALUE));
        TS_ASSERT_EQUALS(26, (Stencils::OffsetHelper<Stencils::Moore<3, 1>,  1,  1,  1>::VALUE));

        TS_ASSERT_EQUALS(2, (Stencils::OffsetHelper<Stencils::VonNeumann<1, 1>,  1,  0,  0>::VALUE));

        TS_ASSERT_EQUALS(2, (Stencils::OffsetHelper<Stencils::VonNeumann<1, 2>,  0,  0,  0>::VALUE));

        TS_ASSERT_EQUALS(0, (Stencils::OffsetHelper<Stencils::VonNeumann<2, 1>,  0, -1,  0>::VALUE));
        TS_ASSERT_EQUALS(1, (Stencils::OffsetHelper<Stencils::VonNeumann<2, 1>, -1,  0,  0>::VALUE));
        TS_ASSERT_EQUALS(2, (Stencils::OffsetHelper<Stencils::VonNeumann<2, 1>,  0,  0,  0>::VALUE));
        TS_ASSERT_EQUALS(4, (Stencils::OffsetHelper<Stencils::VonNeumann<2, 1>,  0,  1,  0>::VALUE));

        TS_ASSERT_EQUALS(6, (Stencils::OffsetHelper<Stencils::VonNeumann<2, 2>,  0,  0,  0>::VALUE));
        TS_ASSERT_EQUALS(2, (Stencils::OffsetHelper<Stencils::VonNeumann<2, 2>,  0, -1,  0>::VALUE));
        TS_ASSERT_EQUALS(7, (Stencils::OffsetHelper<Stencils::VonNeumann<2, 2>,  1,  0,  0>::VALUE));

        TS_ASSERT_EQUALS(0, (Stencils::OffsetHelper<Stencils::VonNeumann<3, 1>,  0,  0, -1>::VALUE));
        TS_ASSERT_EQUALS(2, (Stencils::OffsetHelper<Stencils::VonNeumann<3, 1>, -1,  0,  0>::VALUE));
        TS_ASSERT_EQUALS(3, (Stencils::OffsetHelper<Stencils::VonNeumann<3, 1>,  0,  0,  0>::VALUE));
        TS_ASSERT_EQUALS(5, (Stencils::OffsetHelper<Stencils::VonNeumann<3, 1>,  0,  1,  0>::VALUE));

        TS_ASSERT_EQUALS(16, (Stencils::OffsetHelper<Stencils::VonNeumann<3, 2>,  0,  1,  0>::VALUE));
    }

    void testVonNeumannDimDelta()
    {

        TS_ASSERT_EQUALS( 0, (Stencils::VonNeumannDimDelta<2, 1, 0>::VALUE));
        TS_ASSERT_EQUALS( 2, (Stencils::VonNeumannDimDelta<2, 1, 1>::VALUE));

        TS_ASSERT_EQUALS(-2, (Stencils::VonNeumannDimDelta<2, 1, -1>::VALUE));
        TS_ASSERT_EQUALS(-4, (Stencils::VonNeumannDimDelta<2, 2, -1>::VALUE));
        TS_ASSERT_EQUALS(-6, (Stencils::VonNeumannDimDelta<2, 2, -2>::VALUE));

        TS_ASSERT_EQUALS( 0, (Stencils::VonNeumannDimDelta<2, 2, 0>::VALUE));
        TS_ASSERT_EQUALS( 4, (Stencils::VonNeumannDimDelta<2, 2, 1>::VALUE));
        TS_ASSERT_EQUALS( 6, (Stencils::VonNeumannDimDelta<2, 2, 2>::VALUE));

        TS_ASSERT_EQUALS( 0, (Stencils::VonNeumannDimDelta<2, 3, 0>::VALUE));
        TS_ASSERT_EQUALS( 6, (Stencils::VonNeumannDimDelta<2, 3, 1>::VALUE));
        TS_ASSERT_EQUALS(10, (Stencils::VonNeumannDimDelta<2, 3, 2>::VALUE));
        TS_ASSERT_EQUALS(12, (Stencils::VonNeumannDimDelta<2, 3, 3>::VALUE));

        TS_ASSERT_EQUALS( 0, (Stencils::VonNeumannDimDelta<2, 4, 0>::VALUE));
        TS_ASSERT_EQUALS( 8, (Stencils::VonNeumannDimDelta<2, 4, 1>::VALUE));
        TS_ASSERT_EQUALS(14, (Stencils::VonNeumannDimDelta<2, 4, 2>::VALUE));
        TS_ASSERT_EQUALS(18, (Stencils::VonNeumannDimDelta<2, 4, 3>::VALUE));
        TS_ASSERT_EQUALS(20, (Stencils::VonNeumannDimDelta<2, 4, 4>::VALUE));

        TS_ASSERT_EQUALS( 0, (Stencils::VonNeumannDimDelta<3, 1, 0>::VALUE));
        TS_ASSERT_EQUALS( 0, (Stencils::VonNeumannDimDelta<3, 2, 0>::VALUE));
        TS_ASSERT_EQUALS( 0, (Stencils::VonNeumannDimDelta<3, 2, 0>::VALUE));
        TS_ASSERT_EQUALS( 0, (Stencils::VonNeumannDimDelta<3, 4, 0>::VALUE));

        TS_ASSERT_EQUALS( 3, (Stencils::VonNeumannDimDelta<3, 1, 1>::VALUE));
        TS_ASSERT_EQUALS( 9, (Stencils::VonNeumannDimDelta<3, 2, 1>::VALUE));
        TS_ASSERT_EQUALS(19, (Stencils::VonNeumannDimDelta<3, 3, 1>::VALUE));
        TS_ASSERT_EQUALS(33, (Stencils::VonNeumannDimDelta<3, 4, 1>::VALUE));

        TS_ASSERT_EQUALS(12, (Stencils::VonNeumannDimDelta<3, 2, 2>::VALUE));
        TS_ASSERT_EQUALS(28, (Stencils::VonNeumannDimDelta<3, 3, 2>::VALUE));
        TS_ASSERT_EQUALS(52, (Stencils::VonNeumannDimDelta<3, 4, 2>::VALUE));

        TS_ASSERT_EQUALS(31, (Stencils::VonNeumannDimDelta<3, 3, 3>::VALUE));
        TS_ASSERT_EQUALS(61, (Stencils::VonNeumannDimDelta<3, 4, 3>::VALUE));

        TS_ASSERT_EQUALS(64, (Stencils::VonNeumannDimDelta<3, 4, 4>::VALUE));
    }

    void testRadius()
    {
        TS_ASSERT_EQUALS(1,  (Stencils::Moore<1, 1>::RADIUS));
        TS_ASSERT_EQUALS(1,  (Stencils::Moore<2, 1>::RADIUS));
        TS_ASSERT_EQUALS(1,  (Stencils::Moore<3, 1>::RADIUS));

        TS_ASSERT_EQUALS(2,  (Stencils::Moore<1, 2>::RADIUS));
        TS_ASSERT_EQUALS(2,  (Stencils::Moore<2, 2>::RADIUS));
        TS_ASSERT_EQUALS(2,  (Stencils::Moore<3, 2>::RADIUS));

        TS_ASSERT_EQUALS(0,  (Stencils::VonNeumann<1, 0>::RADIUS));
        TS_ASSERT_EQUALS(2,  (Stencils::VonNeumann<1, 2>::RADIUS));
        TS_ASSERT_EQUALS(4,  (Stencils::VonNeumann<1, 4>::RADIUS));

        TS_ASSERT_EQUALS(1,  (Stencils::Cross<1, 1>::RADIUS));
        TS_ASSERT_EQUALS(2,  (Stencils::Cross<1, 2>::RADIUS));
        TS_ASSERT_EQUALS(3,  (Stencils::Cross<1, 3>::RADIUS));
    }

    void testVolume()
    {
        TS_ASSERT_EQUALS(3,  (Stencils::Moore<1, 1>::VOLUME));
        TS_ASSERT_EQUALS(5,  (Stencils::Moore<1, 2>::VOLUME));
        TS_ASSERT_EQUALS(1,  (Stencils::Moore<2, 0>::VOLUME));
        TS_ASSERT_EQUALS(9,  (Stencils::Moore<2, 1>::VOLUME));
        TS_ASSERT_EQUALS(25, (Stencils::Moore<2, 2>::VOLUME));
        TS_ASSERT_EQUALS(27, (Stencils::Moore<3, 1>::VOLUME));

        TS_ASSERT_EQUALS(1,  (Stencils::VonNeumann<1, 0>::VOLUME));
        TS_ASSERT_EQUALS(3,  (Stencils::VonNeumann<1, 1>::VOLUME));
        TS_ASSERT_EQUALS(5,  (Stencils::VonNeumann<1, 2>::VOLUME));
        TS_ASSERT_EQUALS(5,  (Stencils::VonNeumann<2, 1>::VOLUME));
        TS_ASSERT_EQUALS(13, (Stencils::VonNeumann<2, 2>::VOLUME));
        TS_ASSERT_EQUALS(25, (Stencils::VonNeumann<2, 3>::VOLUME));
        TS_ASSERT_EQUALS(41, (Stencils::VonNeumann<2, 4>::VOLUME));
        TS_ASSERT_EQUALS(7,  (Stencils::VonNeumann<3, 1>::VOLUME));
        TS_ASSERT_EQUALS(25, (Stencils::VonNeumann<3, 2>::VOLUME));

        TS_ASSERT_EQUALS(3,  (Stencils::Cross<1, 1>::VOLUME));
        TS_ASSERT_EQUALS(5,  (Stencils::Cross<1, 2>::VOLUME));
        TS_ASSERT_EQUALS(5,  (Stencils::Cross<2, 1>::VOLUME));
        TS_ASSERT_EQUALS(9,  (Stencils::Cross<2, 2>::VOLUME));
        TS_ASSERT_EQUALS(7,  (Stencils::Cross<3, 1>::VOLUME));
        TS_ASSERT_EQUALS(13, (Stencils::Cross<3, 2>::VOLUME));
    }

    // instead of re-enumerating all indices here we'll use
    // OffsetHelper's property of being the inverse function to
    // Coords to perform a back-to-back test of Coords. This is
    // only to be trusted as OffsetHelper is tested seperately.
    template<class STENCIL, int INDEX>
    class TestStencilCoords
    {
    public:
        typedef typename STENCIL::template Coords<INDEX> RelCoord;

        void operator()() const
        {
            TS_ASSERT_EQUALS(
                INDEX,
                (Stencils::OffsetHelper<STENCIL, RelCoord::X, RelCoord::Y, RelCoord::Z>::VALUE));
        }
    };

    void testCoords()
    {
        TS_ASSERT_EQUALS(Coord<3>(-1, 0, 0), Coord<3>(Stencils::Moore<1, 1>::Coords<0>()));
        TS_ASSERT_EQUALS(Coord<3>( 0, 0, 0), Coord<3>(Stencils::Moore<1, 1>::Coords<1>()));
        TS_ASSERT_EQUALS(Coord<3>( 1, 0, 0), Coord<3>(Stencils::Moore<1, 1>::Coords<2>()));

        TS_ASSERT_EQUALS(Coord<3>(-1, -1, 0), Coord<3>(Stencils::Moore<2, 1>::Coords<0>()));
        TS_ASSERT_EQUALS(Coord<3>( 0, -1, 0), Coord<3>(Stencils::Moore<2, 1>::Coords<1>()));
        TS_ASSERT_EQUALS(Coord<3>( 1, -1, 0), Coord<3>(Stencils::Moore<2, 1>::Coords<2>()));
        TS_ASSERT_EQUALS(Coord<3>(-1,  0, 0), Coord<3>(Stencils::Moore<2, 1>::Coords<3>()));
        TS_ASSERT_EQUALS(Coord<3>( 0,  0, 0), Coord<3>(Stencils::Moore<2, 1>::Coords<4>()));
        TS_ASSERT_EQUALS(Coord<3>( 1,  0, 0), Coord<3>(Stencils::Moore<2, 1>::Coords<5>()));
        TS_ASSERT_EQUALS(Coord<3>(-1,  1, 0), Coord<3>(Stencils::Moore<2, 1>::Coords<6>()));
        TS_ASSERT_EQUALS(Coord<3>( 0,  1, 0), Coord<3>(Stencils::Moore<2, 1>::Coords<7>()));
        TS_ASSERT_EQUALS(Coord<3>( 1,  1, 0), Coord<3>(Stencils::Moore<2, 1>::Coords<8>()));

        TS_ASSERT_EQUALS(Coord<3>(-1, 0, 0), Coord<3>(Stencils::VonNeumann<1, 1>::Coords<0>()));
        TS_ASSERT_EQUALS(Coord<3>( 0, 0, 0), Coord<3>(Stencils::VonNeumann<1, 1>::Coords<1>()));
        TS_ASSERT_EQUALS(Coord<3>( 1, 0, 0), Coord<3>(Stencils::VonNeumann<1, 1>::Coords<2>()));

        TS_ASSERT_EQUALS(Coord<3>( 0, -1, 0), Coord<3>(Stencils::VonNeumann<2, 1>::Coords<0>()));
        TS_ASSERT_EQUALS(Coord<3>(-1,  0, 0), Coord<3>(Stencils::VonNeumann<2, 1>::Coords<1>()));
        TS_ASSERT_EQUALS(Coord<3>( 0,  0, 0), Coord<3>(Stencils::VonNeumann<2, 1>::Coords<2>()));
        TS_ASSERT_EQUALS(Coord<3>( 1,  0, 0), Coord<3>(Stencils::VonNeumann<2, 1>::Coords<3>()));
        TS_ASSERT_EQUALS(Coord<3>( 0,  1, 0), Coord<3>(Stencils::VonNeumann<2, 1>::Coords<4>()));

        Stencils::Repeat<Stencils::Moore<1, 1>::VOLUME, TestStencilCoords, Stencils::Moore<1, 1> >()();
        Stencils::Repeat<Stencils::Moore<2, 1>::VOLUME, TestStencilCoords, Stencils::Moore<2, 1> >()();
        Stencils::Repeat<Stencils::Moore<3, 1>::VOLUME, TestStencilCoords, Stencils::Moore<3, 1> >()();

        Stencils::Repeat<Stencils::Moore<1, 2>::VOLUME, TestStencilCoords, Stencils::Moore<1, 2> >()();
        Stencils::Repeat<Stencils::Moore<2, 2>::VOLUME, TestStencilCoords, Stencils::Moore<2, 2> >()();

        Stencils::Repeat<Stencils::VonNeumann<1, 1>::VOLUME, TestStencilCoords, Stencils::VonNeumann<1, 1> >()();
        Stencils::Repeat<Stencils::VonNeumann<2, 1>::VOLUME, TestStencilCoords, Stencils::VonNeumann<2, 1> >()();
        Stencils::Repeat<Stencils::VonNeumann<3, 1>::VOLUME, TestStencilCoords, Stencils::VonNeumann<3, 1> >()();
    }
};

}
