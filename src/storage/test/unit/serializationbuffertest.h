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

        TS_ASSERT_EQUALS(SerializationBufferType::getData(buf), &buf[0]);

        TS_ASSERT_EQUALS(sizeof(SerializationBufferType::ElementType), sizeof(TestCell<2>));
    }
};

}
