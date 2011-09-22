#include <iostream>
#include <fstream>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/io/ppmwriter.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/simplecellplotter.h>
#include <libgeodecomp/io/tracingwriter.h>


using namespace LibGeoDecomp;

class BuggyCell
{
public:
    typedef Topologies::Cube<2>::Topology Topology;
    friend class BuggyCellToColor;

    static int nanoSteps()
    {
        return 1;
    }

    BuggyCell(const char& _val=0) :
        val(_val)
    {}

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighbors, const unsigned& nanoStep)
    {
        int buf = (neighbors[Coord<2>(0, -1)].val + neighbors[Coord<2>(-1, 0)].val +
                    neighbors[Coord<2>(1,  0)].val + neighbors[Coord<2>( 0, 1)].val);
        val = (buf >> 2) + 1;
    }

private:
    char val;
};

class BuggyCellInitializer : public SimpleInitializer<BuggyCell>
{
public:
    BuggyCellInitializer(std::string infile, const int& steps=10000) :
        SimpleInitializer<BuggyCell>(readDim(infile), steps),
        filename(infile)
    {}

    virtual void grid(GridBase<BuggyCell, 2> *ret)
    {
        CoordBox<2> rect = ret->boundingBox();
        ret->atEdge() = BuggyCell();
        std::string buf;
        int width;
        int height;

        std::cout << "Loading input\n";
        std::ifstream input(filename.c_str());
        input >> buf >> width >> height;

        char c;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                input >> c >> c >> c;
                std::cout << ((c == 0)? " " : "x");
                Coord<2> pos(x, y);
                if (rect.inBounds(pos))
                    ret->at(pos) = BuggyCell(c);
            }
            std::cout << "\n";
        }
        std::cout << "done\n";
    }

private:
    std::string filename;

    Coord<2> readDim(std::string filename)
    {
        std::string buf;
        int width;
        int height;
        std::ifstream input(filename.c_str());
        if (!input.good()) {
            std::cerr << "failed to open input file\n";
            exit(1);
        }
        input >> buf >> width >> height;
        return Coord<2>(width, height);
    }
};

class BuggyCellToColor
{
public:
    Color operator()(const BuggyCell& cell)
    {
        char r = ((cell.val >> 5) & 7) * 255 / 7;
        char g = ((cell.val >> 2) & 7) * 255 / 7;
        char b = ((cell.val >> 0) & 3) * 255 / 3;
        return Color(r, g, b);
    }
};

void runSimulation()
{
    int outputFrequency = 1;
    SerialSimulator<BuggyCell> sim(new BuggyCellInitializer("pic9_evil_smiley.ppm"));
    new PPMWriter<BuggyCell, SimpleCellPlotter<BuggyCell, BuggyCellToColor> >(
        "./smiley", 
        &sim,
        outputFrequency,
        8,
        8);
    new TracingWriter<BuggyCell>(&sim, outputFrequency);

    sim.run();
}

int main(int, char *[])
{
    runSimulation();
    return 0;
}
