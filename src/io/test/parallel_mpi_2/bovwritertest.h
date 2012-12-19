#include <boost/filesystem.hpp>
#include <cxxtest/TestSuite.h>

#include <libgeodecomp/io/bovwriter.h>
#include <libgeodecomp/io/testinitializer.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>
#include <libgeodecomp/parallelization/hiparsimulator/partitions/zcurvepartition.h>

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

    SuperVector<std::string> files;

    void setUp()
    {
        files.clear();
    }

    void tearDown()
    {
        for (int i = 0; i < files.size(); ++i) {
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

            for (int i = 0; i < files.size(); ++i) {
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

    // fixme: move this somewhere to hiparsimulator/partitions
    // void testBogus()
    // {
    //     Coord<3> offset(0, 0, 0);
    //     Coord<3> dimensions(200, 200, 200);
    //     Coord<3> bricDim(25, 25, 25);
    //     HiParSimulator::ZCurvePartition<3> partition(offset, dimensions);
    //     Grid<double, Topologies::Cube<3>::Topology> g(dimensions, 1);
    //     int counter = 0;

    //     if (MPILayer().rank() == 0) {
    //         for (HiParSimulator::ZCurvePartition<3>::Iterator i = partition.begin();
    //              i != partition.end();
    //              ++i)
    //             g[*i] = 1.0 * (counter++) / dimensions.prod();

    //         MPI::File file = MPIIO<double, Topologies::Cube<3>::Topology>::openFileForWrite(
    //             "test.bov", MPI::COMM_SELF);
            
    //         Coord<3> bovDim;
    //         bovDim[0] = std::max(1, dimensions[0]);
    //         bovDim[1] = std::max(1, dimensions[1]);
    //         bovDim[2] = std::max(1, dimensions[2]);

    //         std::ostringstream buf;
    //         buf << "TIME: " << 100 << "\n"
    //             << "DATA_FILE: " << "test.data" << "\n"
    //             << "DATA_SIZE: " 
    //             << bovDim.x() << " " << bovDim.y() << " " << bovDim.z() << "\n"
    //             << "DATA_FORMAT: " << "DOUBLE" << "\n"
    //             << "VARIABLE: " << "zc" << "\n"
    //             << "DATA_ENDIAN: LITTLE\n"
    //             << "BRICK_ORIGIN: 0 0 0\n"
    //             << "BRICK_SIZE: " 
    //             << bovDim.x() << " " << bovDim.y() << " " << bovDim.z() << "\n"
    //             << "DIVIDE_BRICK: true\n"
    //             << "DATA_BRICKLETS: " 
    //             << bricDim.x() << " " << bricDim.y() << " " << bricDim.z() << "\n";
    //         std::string s = buf.str();
    //         file.Write(s.c_str(), s.length(), MPI::CHAR);
    //         file.Close();

    //         file = MPIIO<double, Topologies::Cube<3>::Topology>::openFileForWrite(
    //             "test.data", MPI::COMM_SELF);
    //         CoordBox<3> box(offset, dimensions);
    //         for (CoordBox<DIM>::Iterator i = box.begin(); i != box.end(); ++i) {
    //             file.Write(&g[*i], 1, MPI::DOUBLE);
    //         }
    //         file.Close();
    //     }
    // }

    Grid<double, Topologies::Cube<3>::Topology> readGrid(
        std::string filename, 
        Coord<3> dimensions)
    {
        Grid<double, Topologies::Cube<3>::Topology> ret(dimensions);
        MPI::File file = MPIIO<TestCell<3>, Topologies::Cube<3>::Topology>::openFileForRead(
            filename, MPI::COMM_SELF);
        file.Read(&ret[Coord<3>()], dimensions.prod(), MPI::DOUBLE);
        file.Close();
        return ret;
    }
};

}
