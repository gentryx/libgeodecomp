#include <cxxtest/TestSuite.h>
#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/geometry/floatcoord.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class FloatCoordTest : public CxxTest::TestSuite
{
public:
    void testSerializationOfWriterByReference()
    {
        FloatCoord<1> ca1(1.1);
        FloatCoord<2> ca2(2.2, 3.3);
        FloatCoord<3> ca3(4.4, 5.5, 6.6);

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);

        outputArchive << ca1;
        outputArchive << ca2;
        outputArchive << ca3;

        FloatCoord<1> cb1(-1);
        FloatCoord<2> cb2(-1, -1);
        FloatCoord<3> cb3(-1, -1, -1);

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
        boost::shared_ptr<FloatCoord<1> > ca1(new FloatCoord<1>(7.7));
        boost::shared_ptr<FloatCoord<2> > ca2(new FloatCoord<2>(9.9, 8.8));
        boost::shared_ptr<FloatCoord<3> > ca3(new FloatCoord<3>(12.12, 11.11, 10.10));

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);
        outputArchive << ca1;
        outputArchive << ca2;
        outputArchive << ca3;

        boost::shared_ptr<FloatCoord<1> > cb1;
        boost::shared_ptr<FloatCoord<2> > cb2;
        boost::shared_ptr<FloatCoord<3> > cb3;

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
