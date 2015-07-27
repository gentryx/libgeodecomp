#include <cxxtest/TestSuite.h>
#include <libgeodecomp/communication/hpxserialization.h>
#include <libgeodecomp/io/ppmwriter.h>
#include <libgeodecomp/misc/testcell.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class PPMWriterTest : public CxxTest::TestSuite
{
public:
    void testSerializationOfWriterByReference()
    {
        PPMWriter<TestCell<2> > writer1(
            &TestCell<2>::testValue,
            0.0,
            1.0,
            "bingo",
            4711,
            Coord<2>(1, 2));

        PPMWriter<TestCell<2> > writer2(
            &TestCell<2>::cycleCounter,
            unsigned(20),
            unsigned(30),
            "bongo",
            4712,
            Coord<2>(5, 4));

        std::vector<char> buffer;
        hpx::serialization::output_archive outputArchive(buffer);

        outputArchive << writer1;

        hpx::serialization::input_archive inputArchive(buffer);
        inputArchive >> writer2;

        // fixme
    }

};

}
