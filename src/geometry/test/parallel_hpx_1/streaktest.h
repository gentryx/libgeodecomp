#include <cxxtest/TestSuite.h>
#include <libgeodecomp/communication/hpxserialization.h>
#include <libgeodecomp/geometry/streak.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class StreakTest : public CxxTest::TestSuite
{
public:
    void testSerializationOfWriterByReference()
    {
        Streak<1> ca1(Coord<1>(10),         11);
        Streak<2> ca2(Coord<2>(12, 13),     14);
        Streak<3> ca3(Coord<3>(15, 16, 17), 18);

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);

        outputArchive << ca1;
        outputArchive << ca2;
        outputArchive << ca3;

        Streak<1> cb1(Coord<1>(-1),         -1);
        Streak<2> cb2(Coord<2>(-1, -1),     -1);
        Streak<3> cb3(Coord<3>(-1, -1, -1), -1);

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
        boost::shared_ptr<Streak<1> > ca1(new Streak<1>(Coord<1>(20),         21));
        boost::shared_ptr<Streak<2> > ca2(new Streak<2>(Coord<2>(30, 31),     32));
        boost::shared_ptr<Streak<3> > ca3(new Streak<3>(Coord<3>(34, 35, 36), 37));

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);
        outputArchive << ca1;
        outputArchive << ca2;
        outputArchive << ca3;

        boost::shared_ptr<Streak<1> > cb1;
        boost::shared_ptr<Streak<2> > cb2;
        boost::shared_ptr<Streak<3> > cb3;

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
