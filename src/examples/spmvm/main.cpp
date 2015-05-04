#include <cstdlib>
#include <cmath>
#include <map>
#include <fstream>
#include <string>
#include <stdexcept>

#include <libgeodecomp.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/io/asciiwriter.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/storage/unstructuredgrid.h>

using namespace LibGeoDecomp;

class Cell
{
public:
    class API :
        public APITraits::HasUpdateLineX,
        public APITraits::HasUnstructuredTopology,
        public APITraits::HasPredefinedMPIDataType<double>
    {};

    inline explicit Cell(double v = 0) :
        value(v), sum(0)
    {}

    template<typename HOOD_NEW, typename HOOD_OLD>
    static void updateLineX(HOOD_NEW& hoodNew, int indexEnd, HOOD_OLD& hoodOld, unsigned /* nanoStep */)
    {
        for (int i = hoodOld.index(); i < indexEnd; ++i, ++hoodOld) {
            hoodNew[i].sum = 0.;
            for (const auto& j: hoodOld.weights(0)) {
                hoodNew[i].sum += hoodOld[j.first].value * j.second;
            }
        }
    }

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& neighborhood, unsigned /* nanoStep */)
    {
        sum = 0.;
        for (const auto& j: neighborhood.weights(0)) {
            sum += neighborhood[j.first].value * j.second;
        }
    }

    double value;
    double sum;
};

class CellInitializerDiagonal : public SimpleInitializer<Cell>
{
public:
    using SimpleInitializer<Cell>::gridDimensions;

    explicit CellInitializerDiagonal() : SimpleInitializer<Cell>(Coord<1>(100), 1)
    {}

    virtual void grid(GridBase<Cell, 1> *ret)
    {
        // setup diagonal matrix, one neighbor per cell
        UnstructuredGrid<Cell, 1> *grid = dynamic_cast<UnstructuredGrid<Cell, 1> *>(ret);

        std::map<Coord<2>, double> adjacency;

        for (int i = 0; i < 100; ++i) {
            grid->set(Coord<1>(i), Cell(static_cast<double>(i) + 0.1));
            adjacency[Coord<2>(i, i)] = static_cast<double>(i) + 0.1;
        }

        grid->setAdjacency(0, adjacency.begin(), adjacency.end());
    }
};

class CellInitializerMatrix : public SimpleInitializer<Cell>
{
public:
    using SimpleInitializer<Cell>::gridDimensions;

    explicit CellInitializerMatrix(const std::string& rhsFile,
                                   const std::string& matrixFile) :
        SimpleInitializer<Cell>(Coord<1>(1000), 1),
        rhsFile(rhsFile), matrixFile(matrixFile)
    {}

    virtual void grid(GridBase<Cell, 1> *ret)
    {
        // read rhs and matrix from file
        UnstructuredGrid<Cell, 1> *grid = dynamic_cast<UnstructuredGrid<Cell, 1> *>(ret);
        std::map<Coord<2>, double> adjacency;
        std::ifstream rhsIfs;
        std::ifstream matrixIfs;

        rhsIfs.open(rhsFile);
        matrixIfs.open(matrixFile);
        if (rhsIfs.fail() || matrixIfs.fail()) {
            throw std::logic_error("Failed to open files");
        }

        unsigned i = 0;
        double tmp;
        // read rhs vector
        while (rhsIfs >> tmp) {
            grid->set(Coord<1>(i), Cell(tmp));
            ++i;
        }

        // read matrix
        unsigned rows, cols;
        if (!(matrixIfs >> rows >> cols)) {
            throw std::logic_error("Failed to read from matrix file");
        }
        if (rows != cols && rows != i) {
            throw std::logic_error("Dimensions do not match");
        }

        for (unsigned row = 0; row < rows; ++row) {
            for (unsigned col = 0; col < cols; ++col) {
                if (!(matrixIfs >> tmp)) {
                    throw std::logic_error("Failed to read data from matrix");
                }
                adjacency[Coord<2>(row, col)] = tmp;
            }
        }

        rhsIfs.close();
        matrixIfs.close();

        grid->setAdjacency(0, adjacency.begin(), adjacency.end());
    }

private:
    std::string rhsFile;
    std::string matrixFile;
};

void runSimulation(int argc, char *argv[])
{
    SimpleInitializer<Cell> *init;
    int outputFrequency = 1;

    // init
    if (argc > 1) {
        if (argc != 3) {
            throw std::logic_error("Number of arguments is wrong");
        }
        std::string rhsFile = argv[1];
        std::string matrixFile = argv[2];
        init = new CellInitializerMatrix(rhsFile, matrixFile);
    } else {
        init = new CellInitializerDiagonal();
    }
    SerialSimulator<Cell> sim(init);
    sim.addWriter(new TracingWriter<Cell>(outputFrequency, init->maxSteps()));
    sim.addWriter(new ASCIIWriter<Cell>("sum", &Cell::sum, outputFrequency));
    sim.run();
}

int main(int argc, char *argv[])
{
    runSimulation(argc, argv);

    return EXIT_SUCCESS;
}
