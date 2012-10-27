#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/io/ppmwriter.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/simplecellplotter.h>
#include <libgeodecomp/io/tracingwriter.h>

using namespace LibGeoDecomp;

Coord<2> NEIGHBORS[] = {Coord<2>(-1, -1),
                        Coord<2>( 0, -1),
                        Coord<2>( 1, -1),
                        Coord<2>(-1,  0),
                        Coord<2>( 1,  0),
                        Coord<2>(-1,  1),
                        Coord<2>( 0,  1),
                        Coord<2>( 1,  1)};

class Cell
{
    friend class CellToColor;
public:
    typedef Stencils::Moore<2, 1> Stencil;
    typedef Topologies::Cube<2>::Topology Topology;
    class API : public CellAPITraits::Base
    {};

    enum State {EMPTY, FOOD, IDLE_ANT, BUSY_ANT, BARRIER};
    static const double  PI;

    static inline unsigned nanoSteps() 
    { 
        return 3; 
    }

    Cell(State _state=EMPTY) :
        state(_state),
        posX(0),
        posY(0),
        dropFood(false)
    {
        if (isAnt()) 
            randomTurn();
    }

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, const unsigned& nanoStep)
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

const double Cell::PI = 3.14159265;

class CellToColor
{
public:
    Color operator()(const Cell& cell)
    {
        switch(cell.state) {
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
    CellInitializer(
        const Coord<2> _dim = Coord<2>(240, 135),
        const unsigned _steps = 400000) :
        SimpleInitializer<Cell>(_dim, _steps)
    {}

    virtual void grid(GridBase<Cell, 2> *ret)
    {
        ret->atEdge() = Cell(Cell::BARRIER);
        int numAnts =  100;
        int numFood = 500;

        for (int i = 0; i < numFood; ++i) 
            ret->at(randCoord()) = Cell(Cell::FOOD);

        for (int i = 0; i < numAnts; ++i) 
            ret->at(randCoord()) = Cell(Cell::IDLE_ANT);
    }

private:
    Coord<2> randCoord() const
    {
        int x = rand() % gridDimensions().x();
        int y = rand() % gridDimensions().y();
        return Coord<2>(x, y);
    }
};

class AntTracer : public TracingWriter<Cell>
{
public:
    using TracingWriter<Cell>::sim;

    AntTracer(
        MonolithicSimulator<Cell> *sim,
        const int& period) :
        TracingWriter<Cell>(sim, period)
    {}

    virtual void stepFinished()
    {
        TracingWriter<Cell>::stepFinished();

        if ((sim->getStep() % ParallelWriter<Cell>::period) != 0) return;

        const Grid<Cell> *grid = sim->getGrid();
        int numAnts = 0;
        int numFood = 0;
        
        for(unsigned y = 0; y < grid->getDimensions().y(); ++y) {
            for(unsigned x = 0; x < grid->getDimensions().x(); ++x) {
                if ((*grid)[Coord<2>(x, y)].isAnt())
                    ++numAnts;
                if ((*grid)[Coord<2>(x, y)].containsFood())
                    ++numFood;
            }
        }
        std::cout << "  numAnts: " << numAnts << "\n"
                  << "  numFood: " << numFood << "\n";
    }
};

void runSimulation()
{
    srand(1234);
    int outputFrequency = 1;
    SerialSimulator<Cell> sim(new CellInitializer());
    new PPMWriter<Cell, SimpleCellPlotter<Cell, CellToColor> >(
        "./ants", 
        &sim,
        outputFrequency,
        8,
        8);
    new AntTracer(&sim, outputFrequency);

    sim.run();
}

int main(int, char *[])
{
    runSimulation();
    return 0;
}
