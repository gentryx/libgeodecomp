#include <libgeodecomp/io/memorywriter.h>
#include <libgeodecomp/io/mpiiowriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/parallelization/serialsimulator.h>

#include <cxxtest/TestSuite.h>

#include <unistd.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class MPIIOWriterTest : public CxxTest::TestSuite
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
            unlink(files[i].c_str());
        }
    }

    void testBasic()
    {
        MPIIO<TestCell<3> > mpiio;
        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >();
        SerialSimulator<TestCell<3> > sim(init);
        MPIIOWriter<TestCell<3> > *writer = new MPIIOWriter<TestCell<3> >(
            "testmpiiowriter",
            4,
            init->maxSteps());
        MemoryWriter<TestCell<3> > *memoryWriter = new MemoryWriter<TestCell<3> >(4);
        sim.addWriter(writer);
        sim.addWriter(memoryWriter);

        sim.run();

        TS_ASSERT_EQUALS("testmpiiowriter01234.mpiio", writer->filename(1234));

        typedef APITraits::SelectTopology<TestCell<3> >::Value Topology;
        std::vector<Grid<TestCell<3>, Topology> > expected =
            memoryWriter->getGrids();
        std::vector<Grid<TestCell<3>, Topology> > actual;

        for (std::size_t i = 0; i <= 21; i += (i == 20)? 1 : 4) {
            std::string filename = writer->filename(i);
            files.push_back(filename);

            Coord<3> dimensions;
            unsigned step;
            unsigned maxSteps;
            mpiio.readMetadata(&dimensions, &step, &maxSteps, filename);

            Region<3> region;
            region << CoordBox<3>(Coord<3>(), dimensions);
            Grid<TestCell<3>, Topology> buffer(dimensions);
            mpiio.readRegion(
                &buffer,
                filename,
                region);

            TS_ASSERT_EQUALS(step, i);
            TS_ASSERT_EQUALS(maxSteps, unsigned(21));
            actual.push_back(buffer);
        }

        TS_ASSERT_EQUALS(actual.size(), expected.size());
        TS_ASSERT_EQUALS(actual,        expected);
    }
};

}
