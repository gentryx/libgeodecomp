#include <libgeodecomp/geometry/partitions/zcurvepartition.h>
#include <libgeodecomp/io/bovwriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>

#include <boost/filesystem.hpp>
#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class TestValueSelector
{
public:
    typedef double VariableType;

    void operator()(const TestCell<3>& cell, double *storage)
    {
        *storage = cell.testValue;
    }

    static std::string varName()
    {
        return "val";
    }

    static std::string dataFormat()
    {
        return "DOUBLE";
    }

    static int dataComponents()
    {
        return 1;
    }
};

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
            boost::filesystem::remove(files[i]);
        }
    }

    void testBasic()
    {
        TestInitializer<TestCell<3> > *init = new TestInitializer<TestCell<3> >();
        Coord<3> dimensions(init->gridDimensions());

        LoadBalancer *balancer = MPILayer().rank()? 0 : new RandomBalancer;
        StripingSimulator<TestCell<3> > simTest(init, balancer);
        simTest.addWriter(new BOVWriter<TestCell<3>, TestValueSelector>("testbovwriter", 4));
        simTest.run();

        MPILayer().barrier();

        if (MPILayer().rank() == 0) {
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

            for (std::size_t i = 0; i < files.size(); ++i) {
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
        }
    }

    Grid<double, Topologies::Cube<3>::Topology> readGrid(
        std::string filename,
        Coord<3> dimensions)
    {
        Grid<double, Topologies::Cube<3>::Topology> ret(dimensions);
        MPI_File file = MPIIO<TestCell<3>, Topologies::Cube<3>::Topology>::openFileForRead(
            filename, MPI_COMM_SELF);
        MPI_File_read(file, &ret[Coord<3>()], dimensions.prod(), MPI_DOUBLE, MPI_STATUS_IGNORE);
        MPI_File_close(&file);
        return ret;
    }
};

}
