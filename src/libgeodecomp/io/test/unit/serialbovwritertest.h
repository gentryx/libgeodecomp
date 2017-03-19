#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/io/serialbovwriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/stringops.h>
#include <libgeodecomp/misc/tempfile.h>
#include <libgeodecomp/parallelization/serialsimulator.h>

#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class BOVWriterTest : public CxxTest::TestSuite
{
public:

    std::vector<std::string> files;

    void setUp()
    {
        files.clear();
    }

    void tearDown()
    {
        for (std::size_t i = 0; i < files.size(); ++i) {
            TempFile::unlink(files[i]);
        }
    }

    void testBasic()
    {
        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >();
        Coord<3> dimensions(init->gridDimensions());

        SerialSimulator<TestCell<3> > simTest(init);
        simTest.addWriter(
            new SerialBOVWriter<TestCell<3> >(
                Selector<TestCell<3> >(&TestCell<3>::testValue, "val"),
                "testbovwriter",
                4));
        simTest.run();

        Grid<TestCell<3>, Topologies::Cube<3>::Topology> buffer(dimensions);
        Grid<double, Topologies::Cube<3>::Topology> expected(dimensions);
        Grid<double, Topologies::Cube<3>::Topology> actual;

        init->grid(&buffer);
        CoordBox<3> box(Coord<3>(), dimensions);
        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            expected[*i] = buffer[*i].testValue;
        }

        // only test the data files
        files << "testbovwriter.00000.data"
              << "testbovwriter.00004.data"
              << "testbovwriter.00008.data"
              << "testbovwriter.00012.data"
              << "testbovwriter.00016.data"
              << "testbovwriter.00020.data"
              << "testbovwriter.00021.data";

        std::size_t i = 0;
        for (; i < files.size(); ++i) {
            actual = readGrid(files[i], dimensions);
            TS_ASSERT_EQUALS(actual, expected);
        }

        files << "testbovwriter.00000.bov"
              << "testbovwriter.00004.bov"
              << "testbovwriter.00008.bov"
              << "testbovwriter.00012.bov"
              << "testbovwriter.00016.bov"
              << "testbovwriter.00020.bov"
              << "testbovwriter.00021.bov";

        for (; i < files.size(); ++i) {
            std::string actual = readHeader(files[i]);
            std::stringstream expected;

            std::size_t time = i - (files.size() / 2);
            time *= 4;
            if (time == 24) {
                time = 21;
            }

            StringVec tokens = StringOps::tokenize(files[i], ".");
            std::string filename = tokens[0] + "." + tokens[1] + ".data";

            expected << "TIME: " << time << "\n"
                     << "DATA_FILE: " << filename << "\n"
                     << "DATA_SIZE: 13 12 11\n"
                     << "DATA_FORMAT: DOUBLE\n"
                     << "VARIABLE: val\n"
                     << "DATA_ENDIAN: LITTLE\n"
                     << "BRICK_ORIGIN: 0 0 0\n"
                     << "BRICK_SIZE: 13 12 11\n"
                     << "DIVIDE_BRICK: true\n"
                     << "DATA_BRICKLETS: 13 12 11\n"
                     << "DATA_COMPONENTS: 1\n";

            TS_ASSERT_EQUALS(actual, expected.str());
        }
    }

    Grid<double, Topologies::Cube<3>::Topology> readGrid(
        std::string filename,
        Coord<3> dimensions)
    {
        Grid<double, Topologies::Cube<3>::Topology> ret(dimensions);
        std::ifstream file(filename.c_str());
        TS_ASSERT(file);
        file.read(reinterpret_cast<char*>(&ret[Coord<3>()]), dimensions.prod() * sizeof(double));

        return ret;
    }

    std::string readHeader(std::string filename)
    {
        std::string ret;
        std::ifstream file(filename.c_str());
        TS_ASSERT(file);

        while (true) {
            char c;
            file.get(c);
            if (file.eof()) {
                break;
            }
            ret += c;
        }

        return ret;
    }
};

}
