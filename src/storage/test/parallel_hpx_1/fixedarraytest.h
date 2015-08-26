#include <cxxtest/TestSuite.h>
#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/storage/fixedarray.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class FixedArrayTest : public CxxTest::TestSuite
{
public:
    void testSerializationOfWriterByReference()
    {
        FixedArray<int, 20> array1;
        FixedArray<int, 20> array2;
        array1 << 1 << 1 << 2 << 3 << 5;

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);

        outputArchive << array1;

        TS_ASSERT_EQUALS(0, array2.size());

        hpx::serialization::input_archive inputArchive(buffer);
        inputArchive >> array2;

        TS_ASSERT_EQUALS(5, array2.size());
        TS_ASSERT_EQUALS(array1, array2);
    }

    void testSerializationOfWriterViaSharedPointer()
    {
        boost::shared_ptr<FixedArray<int, 20> > array1(new FixedArray<int, 20>);
        boost::shared_ptr<FixedArray<int, 20> > array2;
        *array1 << 8 << 13 << 21 << 34;

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);

        outputArchive << array1;

        hpx::serialization::input_archive inputArchive(buffer);
        inputArchive >> array2;

        TS_ASSERT_EQUALS(4, array2->size());
        TS_ASSERT_EQUALS(*array1, *array2);
    }
};

}
