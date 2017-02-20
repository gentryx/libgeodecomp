#include <cxxtest/TestSuite.h>
#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/misc/sharedptr.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CoordTest : public CxxTest::TestSuite
{
public:
    void testSerializationOfWriterByReference()
    {
        Coord<1> ca1(1);
        Coord<2> ca2(2, 3);
        Coord<3> ca3(4, 5, 6);

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);

        outputArchive << ca1;
        outputArchive << ca2;
        outputArchive << ca3;

        Coord<1> cb1(-1);
        Coord<2> cb2(-1, -1);
        Coord<3> cb3(-1, -1, -1);

        hpx::serialization::input_archive inputArchive(buffer);
        inputArchive >> cb1;
        inputArchive >> cb2;
        inputArchive >> cb3;

        TS_ASSERT_EQUALS(ca1, cb1);
        TS_ASSERT_EQUALS(ca2, cb2);
        TS_ASSERT_EQUALS(ca3, cb3);
    }

    void testSerializationViaSharedPointer()
    {
        SharedPtr<Coord<1> >::Type ca1(new Coord<1>(7));
        SharedPtr<Coord<2> >::Type ca2(new Coord<2>(9, 8));
        SharedPtr<Coord<3> >::Type ca3(new Coord<3>(12, 11, 10));

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);
        outputArchive << ca1;
        outputArchive << ca2;
        outputArchive << ca3;

        SharedPtr<Coord<1> >::Type cb1;
        SharedPtr<Coord<2> >::Type cb2;
        SharedPtr<Coord<3> >::Type cb3;

        hpx::serialization::input_archive inputArchive(buffer);
        inputArchive >> cb1;
        inputArchive >> cb2;
        inputArchive >> cb3;

        TS_ASSERT_EQUALS(*ca1, *cb1);
        TS_ASSERT_EQUALS(*ca2, *cb2);
        TS_ASSERT_EQUALS(*ca3, *cb3);
    }
};

}
