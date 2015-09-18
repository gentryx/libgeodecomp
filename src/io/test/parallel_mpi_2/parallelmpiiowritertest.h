#include <libgeodecomp/io/memorywriter.h>
#include <libgeodecomp/io/parallelmpiiowriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/loadbalancer/randombalancer.h>
#include <libgeodecomp/misc/random.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>

#include <boost/filesystem.hpp>
#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class ParallelMPIIOWriterTest : public CxxTest::TestSuite
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
            boost::filesystem::remove(files[i]);
        }
    }

    void testBasic()
    {
        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >();
        MPIIO<TestCell<3> > mpiio;

        LoadBalancer *balancer = MPILayer().rank()? 0 : new RandomBalancer;
        StripingSimulator<TestCell<3> > simTest(init, balancer);
        ParallelMPIIOWriter<TestCell<3> > *writer = new ParallelMPIIOWriter<TestCell<3> >(
            "testmpiiowriter",
            4,
            init->maxSteps());
        simTest.addWriter(writer);

        simTest.run();

        if (MPILayer().rank() == 0) {
            TestInitializer<TestCell<3> > *init2 = new TestInitializer<TestCell<3> >();

            SerialSimulator<TestCell<3> > simReference(init2);
            MemoryWriter<TestCell<3> > *memoryWriter = new MemoryWriter<TestCell<3> >(4);
            simReference.addWriter(memoryWriter);

            simReference.run();

            TS_ASSERT_EQUALS("testmpiiowriter01234.mpiio", writer->filename(1234));

            typedef APITraits::SelectTopology<TestCell<3> >::Value Topology;
            std::vector<Grid<TestCell<3>, Topology> > expected = memoryWriter->getGrids();
            std::vector<Grid<TestCell<3>, Topology> > actual;

            for (unsigned i = 0; i <= 21; i += (i == 20)? 1 : 4) {
                std::string filename = writer->filename(i);
                files.push_back(filename);

                Coord<3> dimensions;
                unsigned step;
                unsigned maxSteps;
                mpiio.readMetadata(
                    &dimensions, &step, &maxSteps, filename,
                    MPI_COMM_SELF);

                Region<3> region;
                region << CoordBox<3>(Coord<3>(), dimensions);
                Grid<TestCell<3>, Topology> buffer(dimensions);
                mpiio.readRegion(
                    &buffer,
                    filename,
                    region,
                    MPI_COMM_SELF);

                TS_ASSERT_EQUALS(step, i);
                TS_ASSERT_EQUALS(maxSteps, unsigned(21));
                actual.push_back(buffer);
            }

            TS_ASSERT_EQUALS(actual.size(), expected.size());
            TS_ASSERT_EQUALS(actual,        expected);
        }
    }
};

}
