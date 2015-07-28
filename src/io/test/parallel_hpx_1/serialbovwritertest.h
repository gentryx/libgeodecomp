#include <cxxtest/TestSuite.h>
#include <libgeodecomp/communication/hpxserialization.h>
#include <libgeodecomp/io/ppmwriter.h>
#include <libgeodecomp/misc/testcell.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class SerialBOVWriterTest : public CxxTest::TestSuite
{
public:
    void testSerializationOfWriterByReference()
    {
        SerialBOVWriter<TestCell<2> > writer1(
            &TestCell<2>::testValue,
            "bingo",
            4711,
            Coord<3>(1, 2, 3));

        SerialBOVWriter<TestCell<2> > writer2(
            &TestCell<2>::cycleCounter,
            "bongo",
            4712,
            Coord<3>(6, 5, 4));
        TS_ASSERT_EQUALS(Coord<3>(6, 5, 4), writer2.brickletDim);

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);

        outputArchive << writer1;

        hpx::serialization::input_archive inputArchive(buffer);
        inputArchive >> writer2;

        TS_ASSERT_EQUALS(Coord<3>(1, 2, 3), writer2.brickletDim);
    }

    void testSerializationOfWriterViaSharedPtr()
    {
        boost::shared_ptr<Writer<TestCell<2> > > writer1(new SerialBOVWriter<TestCell<2> >(
            &TestCell<2>::testValue,
            "bingo",
            4711,
            Coord<3>(9, 99, 999)));

        boost::shared_ptr<Writer<TestCell<2> > > writer2;

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);

        outputArchive << writer1;

        hpx::serialization::input_archive inputArchive(buffer);
        inputArchive >> writer2;

        TS_ASSERT_EQUALS(
            Coord<3>(9, 99, 999),
            dynamic_cast<SerialBOVWriter<TestCell<2> >*>(&*writer2)->brickletDim);
    }

};

}
