#include <cxxtest/TestSuite.h>
#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/storage/serializationbuffer.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class SerializationBufferTest : public CxxTest::TestSuite
{
public:
    void testTestCell2D()
    {
        typedef SerializationBuffer<TestCell<2> > SerializationBufferType;

        Region<2> region;
        region << Streak<2>(Coord<2>(10, 10), 30)
               << Streak<2>(Coord<2>(15, 11), 50);

        SerializationBufferType::BufferType buf = SerializationBufferType::create(region);
        TS_ASSERT_EQUALS(buf.size(), region.size());
        TS_ASSERT_EQUALS(SerializationBufferType::minimumStorageSize(region), region.size());

        region << Streak<2>(Coord<2>(0, 0), 20);
        SerializationBufferType::resize(&buf, region);
        TS_ASSERT_EQUALS(buf.size(), region.size());

        TS_ASSERT_EQUALS(SerializationBufferType::getData(buf), &buf[0]);

        TS_ASSERT_EQUALS(sizeof(SerializationBufferType::ElementType), sizeof(TestCell<2>));
    }

    void testTestCell3D()
    {
        typedef SerializationBuffer<TestCell<3> > SerializationBufferType;

        Region<3> region;
        region << Streak<3>(Coord<3>(10, 10, 10), 30)
               << Streak<3>(Coord<3>(15, 11, 10), 50)
               << Streak<3>(Coord<3>(10, 11, 15), 90);

        SerializationBufferType::BufferType buf = SerializationBufferType::create(region);
        TS_ASSERT_EQUALS(buf.size(), region.size());
        TS_ASSERT_EQUALS(SerializationBufferType::minimumStorageSize(region), region.size());

        region << Streak<3>(Coord<3>(0, 0, 0), 20);
        SerializationBufferType::resize(&buf, region);
        TS_ASSERT_EQUALS(buf.size(), region.size());

        TS_ASSERT_EQUALS(SerializationBufferType::getData(buf), &buf[0]);

        TS_ASSERT_EQUALS(sizeof(SerializationBufferType::ElementType), sizeof(TestCell<3>));
    }

    void testTestCellSoA()
    {
        typedef SerializationBuffer<TestCellSoA > SerializationBufferType;

        Region<3> region;
        region << Streak<3>(Coord<3>(10, 10, 10), 30)
               << Streak<3>(Coord<3>(15, 11, 10), 50)
               << Streak<3>(Coord<3>(10, 11, 15), 90);

        SerializationBufferType::BufferType buf = SerializationBufferType::create(region);
        std::size_t expectedSize = LibFlatArray::aggregated_member_size<TestCellSoA>::VALUE * region.size();
        TS_ASSERT_EQUALS(buf.size(), expectedSize);
        TS_ASSERT_EQUALS(SerializationBufferType::minimumStorageSize(region), expectedSize);

        region << Streak<3>(Coord<3>(0, 0, 0), 25);
        expectedSize = LibFlatArray::aggregated_member_size<TestCellSoA>::VALUE * region.size();
        SerializationBufferType::resize(&buf, region);
        TS_ASSERT_EQUALS(buf.size(), expectedSize);
        TS_ASSERT_EQUALS(SerializationBufferType::minimumStorageSize(region), expectedSize);

        TS_ASSERT_EQUALS(SerializationBufferType::getData(buf), &buf[0]);

        TS_ASSERT_EQUALS(sizeof(SerializationBufferType::ElementType), sizeof(char));
    }
};

}
