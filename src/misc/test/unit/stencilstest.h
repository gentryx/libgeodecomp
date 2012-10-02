#include <cxxtest/TestSuite.h>
#include <libgeodecomp/misc/stencils.h>

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

    void testSum()
    {
        TS_ASSERT_EQUALS( 1, (Stencils::Sum<Stencils::Moore, 0, 1>::VALUE));
        TS_ASSERT_EQUALS( 4, (Stencils::Sum<Stencils::Moore, 1, 1>::VALUE));
        TS_ASSERT_EQUALS(13, (Stencils::Sum<Stencils::Moore, 2, 1>::VALUE));
    }

    void testOffsetHelper()
    {
        TS_ASSERT_EQUALS( 0, (Stencils::OffsetHelper<Stencils::Moore<1, 1>, -1,  0, 0>::VALUE));

        TS_ASSERT_EQUALS( 4, (Stencils::OffsetHelper<Stencils::Moore<2, 1>,  0,  0, 0>::VALUE));
        TS_ASSERT_EQUALS( 3, (Stencils::OffsetHelper<Stencils::Moore<2, 1>, -1,  0, 0>::VALUE));
        TS_ASSERT_EQUALS( 0, (Stencils::OffsetHelper<Stencils::Moore<2, 1>, -1, -1, 0>::VALUE));
        TS_ASSERT_EQUALS( 3, (Stencils::OffsetHelper<Stencils::Moore<2, 1>, -1,  0, 0>::VALUE));
        TS_ASSERT_EQUALS( 7, (Stencils::OffsetHelper<Stencils::Moore<2, 1>,  0,  1, 0>::VALUE));
        TS_ASSERT_EQUALS( 8, (Stencils::OffsetHelper<Stencils::Moore<2, 1>,  1,  1, 0>::VALUE));

        TS_ASSERT_EQUALS( 0, (Stencils::OffsetHelper<Stencils::Moore<3, 1>, -1, -1, -1>::VALUE));
        TS_ASSERT_EQUALS(10, (Stencils::OffsetHelper<Stencils::Moore<3, 1>,  0, -1,  0>::VALUE));
        TS_ASSERT_EQUALS(13, (Stencils::OffsetHelper<Stencils::Moore<3, 1>,  0,  0,  0>::VALUE));
        TS_ASSERT_EQUALS(25, (Stencils::OffsetHelper<Stencils::Moore<3, 1>,  0,  1,  1>::VALUE));
        TS_ASSERT_EQUALS(26, (Stencils::OffsetHelper<Stencils::Moore<3, 1>,  1,  1,  1>::VALUE));

        TS_ASSERT_EQUALS(2, (Stencils::OffsetHelper<Stencils::VonNeumann<1, 1>,  1,  0,  0>::VALUE));     

        TS_ASSERT_EQUALS(0, (Stencils::OffsetHelper<Stencils::VonNeumann<2, 1>,  0, -1,  0>::VALUE));     
        TS_ASSERT_EQUALS(1, (Stencils::OffsetHelper<Stencils::VonNeumann<2, 1>, -1,  0,  0>::VALUE));     
        TS_ASSERT_EQUALS(2, (Stencils::OffsetHelper<Stencils::VonNeumann<2, 1>,  0,  0,  0>::VALUE));     
        TS_ASSERT_EQUALS(4, (Stencils::OffsetHelper<Stencils::VonNeumann<2, 1>,  0,  1,  0>::VALUE));     

        TS_ASSERT_EQUALS(0, (Stencils::OffsetHelper<Stencils::VonNeumann<3, 1>,  0,  0, -1>::VALUE));     
        TS_ASSERT_EQUALS(2, (Stencils::OffsetHelper<Stencils::VonNeumann<3, 1>, -1,  0,  0>::VALUE));     
        TS_ASSERT_EQUALS(3, (Stencils::OffsetHelper<Stencils::VonNeumann<3, 1>,  0,  0,  0>::VALUE));     
        TS_ASSERT_EQUALS(5, (Stencils::OffsetHelper<Stencils::VonNeumann<3, 1>,  0,  1,  0>::VALUE));     
    }

    void testVolume()
    {
        TS_ASSERT_EQUALS(3,  (Stencils::Moore<1, 1>::VOLUME));
        TS_ASSERT_EQUALS(5,  (Stencils::Moore<1, 2>::VOLUME));
        TS_ASSERT_EQUALS(9,  (Stencils::Moore<2, 1>::VOLUME));
        TS_ASSERT_EQUALS(25, (Stencils::Moore<2, 2>::VOLUME));
        TS_ASSERT_EQUALS(27, (Stencils::Moore<3, 1>::VOLUME));

        TS_ASSERT_EQUALS(3,  (Stencils::VonNeumann<1, 1>::VOLUME));
        TS_ASSERT_EQUALS(5,  (Stencils::VonNeumann<1, 2>::VOLUME));
        TS_ASSERT_EQUALS(5,  (Stencils::VonNeumann<2, 1>::VOLUME));
        TS_ASSERT_EQUALS(13, (Stencils::VonNeumann<2, 2>::VOLUME));
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
        Stencils::Repeat<Stencils::VonNeumann<1, 1>::VOLUME, TestStencilCoords, Stencils::VonNeumann<1, 1> >()();
        Stencils::Repeat<Stencils::VonNeumann<2, 1>::VOLUME, TestStencilCoords, Stencils::VonNeumann<2, 1> >()();
        Stencils::Repeat<Stencils::VonNeumann<3, 1>::VOLUME, TestStencilCoords, Stencils::VonNeumann<3, 1> >()();
    }
};

}
