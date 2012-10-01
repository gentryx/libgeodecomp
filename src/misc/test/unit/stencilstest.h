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

    void testCells()
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

    void testCoords()
    {
        // TS_ASSERT_EQUALS(
    }
};

}
