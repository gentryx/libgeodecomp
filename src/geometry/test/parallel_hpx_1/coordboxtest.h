#include <cxxtest/TestSuite.h>
#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/geometry/coordbox.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CoordBoxTest : public CxxTest::TestSuite
{
public:
    void testSerializationOfWriterByReference()
    {
        CoordBox<1> ca1(Coord<1>(1),       Coord<1>(2));
        CoordBox<2> ca2(Coord<2>(3, 4),    Coord<2>(5, 6));
        CoordBox<3> ca3(Coord<3>(7, 8, 9), Coord<3>(10, 11, 12));

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);

        outputArchive << ca1;
        outputArchive << ca2;
        outputArchive << ca3;

        CoordBox<1> cb1(Coord<1>(-1),         Coord<1>(-1));
        CoordBox<2> cb2(Coord<2>(-1, -1),     Coord<2>(-1, -1));
        CoordBox<3> cb3(Coord<3>(-1, -1, -1), Coord<3>(-1, -1, -1));

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
        boost::shared_ptr<CoordBox<1> > ca1(new CoordBox<1>(Coord<1>(1),       Coord<1>(2)));
        boost::shared_ptr<CoordBox<2> > ca2(new CoordBox<2>(Coord<2>(3, 4),    Coord<2>(5, 6)));
        boost::shared_ptr<CoordBox<3> > ca3(new CoordBox<3>(Coord<3>(7, 8, 9), Coord<3>(10, 11, 12)));

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);
        outputArchive << ca1;
        outputArchive << ca2;
        outputArchive << ca3;

        boost::shared_ptr<CoordBox<1> > cb1;
        boost::shared_ptr<CoordBox<2> > cb2;
        boost::shared_ptr<CoordBox<3> > cb3;

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
