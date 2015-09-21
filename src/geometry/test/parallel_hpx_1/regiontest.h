#include <cxxtest/TestSuite.h>
#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/geometry/region.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class RegionTest : public CxxTest::TestSuite
{
public:
    void testSerializationOfWriterByReference()
    {
        Region<1> ca1;
        Region<2> ca2;
        Region<3> ca3;

        ca1 << Streak<1>(Coord<1>(10),         11);

        ca2 << Streak<2>(Coord<2>(12, 13),     14);
        ca2 << Streak<2>(Coord<2>(15, 16),     17);

        ca3 << Streak<3>(Coord<3>(18, 19, 20), 21);
        ca3 << Streak<3>(Coord<3>(22, 19, 24), 25);
        ca3 << Streak<3>(Coord<3>(26, 27, 28), 29);

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);

        outputArchive << ca1;
        outputArchive << ca2;
        outputArchive << ca3;
        Region<1> cb1;
        Region<2> cb2;
        Region<3> cb3;

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
        boost::shared_ptr<Region<1> > ca1(new Region<1>);
        boost::shared_ptr<Region<2> > ca2(new Region<2>);
        boost::shared_ptr<Region<3> > ca3(new Region<3>);

        *ca1 << Streak<1>(Coord<1>(210),           211);
        *ca2 << Streak<2>(Coord<2>(212, 213),      214);
        *ca2 << Streak<2>(Coord<2>(215, 216),      217);

        *ca3 << Streak<3>(Coord<3>(218, 219, 220), 221);
        *ca3 << Streak<3>(Coord<3>(222, 219, 224), 225);
        *ca3 << Streak<3>(Coord<3>(226, 227, 228), 229);

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);
        outputArchive << ca1;
        outputArchive << ca2;
        outputArchive << ca3;

        boost::shared_ptr<Region<1> > cb1;
        boost::shared_ptr<Region<2> > cb2;
        boost::shared_ptr<Region<3> > cb3;

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
