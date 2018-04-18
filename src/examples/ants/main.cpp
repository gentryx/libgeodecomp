#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/io/ppmwriter.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/geometry/adjacencymanufacturer.h>
#include <libgeodecomp/geometry/adjacency.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/geometry/regionbasedadjacency.h>
#include <libgeodecomp/misc/sharedptr.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/random.h>
#include <libgeodecomp/storage/gridbase.h>
#include <libgeodecomp/geometry/regionbasedadjacency.h>
#include <libgeodecomp/io/simplecellplotter.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libflatarray/macros.hpp>

using namespace LibGeoDecomp;

Coord<2> NEIGHBORS[] = {Coord<2>(-1, -1),
                        Coord<2>( 0, -1),
                        Coord<2>( 1, -1),
                        Coord<2>(-1,  0),
                        Coord<2>( 1,  0),
                        Coord<2>(-1,  1),
                        Coord<2>( 0,  1),
                        Coord<2>( 1,  1)};

// don't warn about padding here, we just don't care.
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4820 )
#endif

class Cell
{
public:
    friend void runSimulation();

    class API :
        public APITraits::HasNanoSteps<3>
    {};

    enum State {EMPTY, FOOD, IDLE_ANT, BUSY_ANT, BARRIER};
    static const double PI;

    explicit Cell(State state=EMPTY) :
        state(state),
        posX(0),
        posY(0),
        dropFood(false)
    {
        if (isAnt())
            randomTurn();
    }

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, unsigned nanoStep)
    {
        *this = neighborhood[Coord<2>(0, 0)];

        // determine target cell
        if (nanoStep == 0) {
            incoming = 0;
            target = Coord<2>(0, 0);

            if (isAnt()) {
                posX += cos(dir * 2 * PI / 360);
                posY += sin(dir * 2 * PI / 360);

                target = Coord<2>((int)posX, (int)posY);

                if (target != Coord<2>(0, 0)) {
                    Cell targetCell = neighborhood[target];
                    if ((targetCell.state == EMPTY) ||
                        (targetCell.state == FOOD &&
                         state == IDLE_ANT)) {
                        // move there
                    } else {
                        if (targetCell.state == FOOD && state == BUSY_ANT) {
                            dropFood = true;
                        }
                        randomTurn();
                    }
                }
            }
        }

        // count incoming ants
        if (nanoStep == 1) {
            for (int i = 0; i < 8; ++i) {
                Coord<2> neigh = NEIGHBORS[i];
                if (neigh == -neighborhood[neigh].target)
                    ++incoming;
            }
        }

        // move
        if (nanoStep == 2) {
            if (incoming == 1) {
                for (int i = 0; i < 8; ++i) {
                    Coord<2> neigh = NEIGHBORS[i];
                    if (neigh == -neighborhood[neigh].target) {
                        *this = neighborhood[neigh];
                        posX += neigh.x();
                        posY += neigh.y();
                        if (dropFood) {
                            state = IDLE_ANT;
                            dropFood = false;
                        }

                        if (neighborhood[Coord<2>(0, 0)].state == FOOD) {
                            state = BUSY_ANT;
                            dropFood = false;
                            randomTurn();
                        }
                    }
                }
            } else {
                if (isAnt() && target != Coord<2>(0, 0)) {
                    if (neighborhood[target].incoming == 1) {
                        *this = Cell(dropFood? FOOD :EMPTY);
                    } else {
                        randomTurn();
                    }
                }
            }

        }
    }

    bool isAnt() const
    {
        return state == IDLE_ANT || state == BUSY_ANT;
    }

    bool containsFood() const
    {
        return state == FOOD || state == BUSY_ANT;
    }

private:
    State state;
    double dir;
    double posX;
    double posY;
    int incoming;
    Coord<2> target;
    bool dropFood;

    void randomTurn()
    {
        dir = rand() % 360;
        posX = 0;
        posY = 0;
        target = Coord<2>(0, 0);
    }

};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif


const double Cell::PI = 3.14159265;

class CellToColor
{
public:
    Color operator[](const Cell::State& state) const
    {
        switch (state) {
        case Cell::EMPTY:
            return Color::BLACK;
        case Cell::FOOD:
            return Color::YELLOW;
        case Cell::IDLE_ANT:
            return Color::BLUE;
        case Cell::BUSY_ANT:
            return Color::MAGENTA;
        case Cell::BARRIER:
            return Color::GREEN;
        default:
            return Color::WHITE;
        }
    }
};

class CellInitializer : public SimpleInitializer<Cell>
{
public:
    explicit CellInitializer(
        const Coord<2> dim = Coord<2>(240, 135),
        const unsigned steps = 400000) :
        SimpleInitializer<Cell>(dim, steps)
    {}

    virtual void grid(GridBase<Cell, 2> *ret)
    {
        ret->setEdge(Cell(Cell::BARRIER));
        int numAnts =  100;
        int numFood = 500;

        for (int i = 0; i < numFood; ++i) {
            ret->set(randCoord(), Cell(Cell::FOOD));
        }

        for (int i = 0; i < numAnts; ++i) {
            ret->set(randCoord(), Cell(Cell::IDLE_ANT));
        }
    }

private:
    Coord<2> randCoord() const
    {
        int x = rand() % gridDimensions().x();
        int y = rand() % gridDimensions().y();
        return Coord<2>(x, y);
    }
};

#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4626 5027 )
#endif

class AntTracer : public TracingWriter<Cell>
{
public:
    AntTracer(
        const unsigned period,
        const unsigned maxSteps ) :
        TracingWriter<Cell>(period, maxSteps)
    {}

    void stepFinished(const WriterGridType& grid, unsigned step, WriterEvent event)
    {
        TracingWriter<Cell>::stepFinished(grid, step, event);

        if ((step % ParallelWriter<Cell>::period) != 0) {
            return;
        }

        int numAnts = 0;
        int numFood = 0;

        Coord<2> dim = grid.dimensions();

        for(int y = 0; y < dim.y(); ++y) {
            for(int x = 0; x < dim.x(); ++x) {
                if (grid.get(Coord<2>(x, y)).isAnt()) {
                    ++numAnts;
                }
                if (grid.get(Coord<2>(x, y)).containsFood()) {
                    ++numFood;
                }
            }
        }

        std::cout << "  numAnts: " << numAnts << "\n"
                  << "  numFood: " << numFood << "\n";
    }
};

void runSimulation()
{
    srand(1234);
    unsigned outputFrequency = 1;
    CellInitializer *init = new CellInitializer();
    SerialSimulator<Cell> sim(init);
    sim.addWriter(
        new PPMWriter<Cell>(
            &Cell::state,
            CellToColor(),
            "./ants",
            outputFrequency,
            Coord<2>(8, 8)));
    sim.addWriter(
        new AntTracer(
            outputFrequency,
            init->maxSteps()));

    sim.run();
}

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

int main(int /* argc */, char** /* argv */)
{
    runSimulation();
    return 0;
}

LIBFLATARRAY_DISABLE_SYSTEM_HEADER_WARNINGS_EOF
