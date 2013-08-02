#include <cmath>
/**
 * We need to include typemaps first to avoid problems with Intel
 * MPI's C++ bindings (which may collide with stdio.h's SEEK_SET,
 * SEEK_CUR etc.).
 */
#include <libgeodecomp/mpilayer/typemaps.h>
#include <libgeodecomp/io/bovwriter.h>
#include <libgeodecomp/io/parallelmpiiowriter.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/loadbalancer/noopbalancer.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>

using namespace LibGeoDecomp;

#define LID 1
#define WING 2
#define SETUP 1

// fixme: use dX for... anything
// fixme: tune pressure speed, pressure diffusion, driver velocity, influence factor?
// fixme: don't equalize pressure on lid?
const double dX = 1.0;
const double dT = 1.0;

#if SETUP==LID
const int MAX_X = 512;
const int MAX_Y = 512;
const double FLOW_DIFFUSION = 0.1 * dT;
const double PRESSURE_DIFFUSION = 0.1 * dT;
#endif

#if SETUP==WING
const int MAX_X = 2048;
const int MAX_Y = 2048;
const double FLOW_DIFFUSION = 0.1 * dT;
const double PRESSURE_DIFFUSION = 0.1 * dT;
#endif

const double PRESSURE_SPEED = 0.1;

const double FRICTION = 0.00015;
const double DRIVER_VELOCITY_X = 3.0;
const double DRIVER_VELOCITY_Y = 0.0;

enum State {LIQUID=0, SLIP=1, SOLID=2, CONST=3};

Coord<2> NEIGHBOR_COORDS[] = {Coord<2>(-1, -1),
                              Coord<2>( 0, -1),
                              Coord<2>( 1, -1),
                              Coord<2>(-1,  0),
                              Coord<2>( 1,  0),
                              Coord<2>(-1,  1),
                              Coord<2>( 0,  1),
                              Coord<2>( 1,  1)};

const double DIAG = 0.707107;

double PERPENDICULAR_DIRS[][2] = {{DIAG, -DIAG},
                                  {1, 0},
                                  {DIAG, DIAG},
                                  {0, -1},
                                  {0, 1},
                                  {-DIAG, -DIAG},
                                  {-1, 0},
                                  {-DIAG, DIAG}};

const double INFLUENCE_FACTOR = 0.04;
// fixme: reduce this to 1 influence?
double INFLUENCES[] =
    {INFLUENCE_FACTOR * FLOW_DIFFUSION, FLOW_DIFFUSION, INFLUENCE_FACTOR * FLOW_DIFFUSION,
     FLOW_DIFFUSION, FLOW_DIFFUSION,
     INFLUENCE_FACTOR * FLOW_DIFFUSION, FLOW_DIFFUSION, INFLUENCE_FACTOR * FLOW_DIFFUSION};

double LENGTHS[] =
    {DIAG, 1.0, DIAG,
     1.0, 1.0,
     DIAG, 1.0, DIAG};

class Cell
{
public:
    typedef Stencils::Moore<2, 1> Stencil;
    typedef Topologies::Cube<2>::Topology Topology;

    class API : public CellAPITraits::Base
    {};

    static int nanoSteps()
    {
        return 1;
    }

    Cell(const State& state = LIQUID,
         const double& quantity = 0,
         const double& velocityX = 0,
         const double& velocityY = 0) :
        state(state),
        quantity(quantity),
        velocityX(velocityX),
        velocityY(velocityY)
    {}

    template<class COORD_MAP>
    void update(const COORD_MAP& neighbors, const int& nanoStep)
    {
        *this = neighbors[Coord<2>()];

        if (state == SOLID)
            return;

        diffuse(neighbors);
    }

    static void flux(const Cell& from, const Cell& to, const int& i,
                     double *fluxFlow, double *fluxPressure)
    {
        if (from.state == SOLID || to.state == SOLID) {
            *fluxFlow = 0;
            *fluxPressure = 0;
            return;
        }

        const Coord<2>& dir = NEIGHBOR_COORDS[i];
        const double& influence = INFLUENCES[i];

        *fluxFlow = (dir.x() * from.velocityX + dir.y() * from.velocityY) *
            influence * from.quantity;
        *fluxFlow = std::max(0.0, *fluxFlow);

        *fluxPressure = 0;
        if (from.quantity > to.quantity)
            *fluxPressure = (from.quantity - to.quantity) * PRESSURE_DIFFUSION;


    }

    template<class COORD_MAP>
    static void addFlowFromNeighbor(
        const Cell& oldSelf,
        const int& i,
        const COORD_MAP& neighbors,
        double *fluxVelocityX,
        double *fluxVelocityY,
        double *newQuantity)
    {
        const Coord<2>& dir = NEIGHBOR_COORDS[i];
        const Cell& other = neighbors[dir];
        const double& length = LENGTHS[i];

        double fluxFlow;
        double fluxPressure;
        flux(other, oldSelf, 7 - i, &fluxFlow, &fluxPressure);

        if (fluxFlow == 0 && fluxPressure == 0)
            return;

        double totalFlow = fluxFlow + fluxPressure;
        *fluxVelocityX += totalFlow * other.velocityX;
        *fluxVelocityY += totalFlow * other.velocityY;
        *newQuantity += totalFlow;

        double pressureCoefficient = length * PRESSURE_SPEED * fluxPressure;
        *fluxVelocityX += -dir.x() * pressureCoefficient;
        *fluxVelocityY += -dir.y() * pressureCoefficient;
    }

    template<class COORD_MAP>
    static void removeFlowToNeighbor(
        const Cell& oldSelf,
        const int& i,
        const COORD_MAP& neighbors,
        double *newQuantity)
    {
        const Coord<2>& dir = NEIGHBOR_COORDS[i];
        const Cell& other = neighbors[dir];
        double fluxFlow;
        double fluxPressure;
        flux(oldSelf, other, i, &fluxFlow, &fluxPressure);

        *newQuantity -= fluxFlow;
        *newQuantity -= fluxPressure;
    }

    template<class COORD_MAP>
    void diffuse(const COORD_MAP& neighbors)
    {
        double fluxVelocityX = 0;
        double fluxVelocityY = 0;
        double newQuantity = quantity;
        const Cell& oldSelf = neighbors[Coord<2>()];

        for (int i = 0; i < 8; ++i)
            addFlowFromNeighbor(oldSelf, i, neighbors,
                                &fluxVelocityX, &fluxVelocityY, &newQuantity);

        double velocityCoeff = (state == SLIP) ? (1 - FRICTION) : 1.0;
        velocityCoeff /= newQuantity;
        if (state != CONST) {
            velocityX = (quantity * velocityX + fluxVelocityX) * velocityCoeff;
            velocityY = (quantity * velocityY + fluxVelocityY) * velocityCoeff;
        }

        for (int i = 0; i < 8; ++i)
            removeFlowToNeighbor(oldSelf, i, neighbors, &newQuantity);

        if (newQuantity < 0) {
            std::cout << "ohoh\n"
                      << "oldSelf:\n"
                      << "  quantity = " << oldSelf.quantity << "\n"
                      << "  velocityX = " << oldSelf.velocityX << "\n"
                      << "  velocityY = " << oldSelf.velocityY << "\n"
                      << "newSelf:\n"
                      << "  quantity = " << newQuantity << "\n"
                      << "  velocityX = " << velocityX << "\n"
                      << "  velocityY = " << velocityY << "\n\n";

            for (int i = 0; i < 8; ++i) {
                double fluxVelocityX = 0;
                double fluxVelocityY = 0;
                double addQuantity = 0;
                double removeQuantity = 0;
                addFlowFromNeighbor(oldSelf, i, neighbors,
                                    &fluxVelocityX, &fluxVelocityY, &addQuantity);
                removeFlowToNeighbor(oldSelf, i, neighbors, &removeQuantity);
                std::cout << "i: " << i << "\n"
                          << "  fluxVelocityX: " << fluxVelocityX << "\n"
                          << "  fluxVelocityY: " << fluxVelocityY << "\n"
                          << "  addQuantity: " << addQuantity << "\n"
                          << "  removeQuantity: " << removeQuantity << "\n";
            }

            throw std::logic_error("negative quantity, unstable simulation!");
        }

        quantity = newQuantity;
    }

    State state;
    double quantity;
    double velocityX;
    double velocityY;
};

class QuantitySelector
{
public:
    typedef double VariableType;

    static std::string varName()
    {
        return "quantity";
    }

    static std::string dataFormat()
    {
        return "DOUBLE";
    }

    static int dataComponents()
    {
        return 1;
    }

    void operator()(const Cell& cell, double *storage)
    {
        *storage = cell.quantity;
    }
};

class VelocitySelector
{
public:
    typedef double VariableType;

    static std::string varName()
    {
        return "velocity";
    }

    static std::string dataFormat()
    {
        return "DOUBLE";
    }

    static int dataComponents()
    {
        return 3;
    }

    void operator()(const Cell& cell, double *storage)
    {
        storage[0] = cell.velocityX;
        storage[1] = cell.velocityY;
        storage[2] = 0;
    }
};

class AeroInitializer : public LibGeoDecomp::SimpleInitializer<Cell>
{
public:
    using LibGeoDecomp::SimpleInitializer<Cell>::dimensions;

    AeroInitializer(
        const Coord<2>& dim,
        const unsigned& steps) :
        SimpleInitializer<Cell>(dim, steps)
    {}

    virtual void grid(GridBase<Cell, 2> *grid)
    {
        CoordBox<2> box = grid->boundingBox();
        grid->setEdge(Cell(SOLID));

        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            grid->set(*i, Cell(LIQUID, 1));
        }

#if SETUP==LID
        addLid(grid);
#endif

#if SETUP==WING
        addInletOutlet(grid);
        // fixme: make this configurable by command line
        //        addWing(grid);
#endif
    }

    void addLid(GridBase<Cell, 2> *grid)
    {
        CoordBox<2> box = grid->boundingBox();

        Cell driverCell(CONST, 1, DRIVER_VELOCITY_X, DRIVER_VELOCITY_Y);
        Cell slipCell(SLIP, 1);

        for (int y = 0; y < dimensions.y(); ++y) {
            Coord<2> c1(0, y);
            Coord<2> c2(dimensions.x() - 1, y);
            if (box.inBounds(c1)) {
                grid->set(c1, slipCell);
            }
            if (box.inBounds(c2)) {
                grid->set(c2, slipCell);
            }
        }

        for (int x = 0; x < dimensions.x(); ++x) {
            Coord<2> c(x, 0);
            if (box.inBounds(c)) {
                grid->set(c, slipCell);
            }
        }

        for (int x = 1; x < dimensions.x() - 1; ++x) {
            Coord<2> c(x, dimensions.y() - 1);
            if (box.inBounds(c)) {
                grid->set(c, driverCell);
            }
        }
    }

    void addInletOutlet(GridBase<Cell, 2> *grid)
    {
        CoordBox<2> box = grid->boundingBox();

        Cell driverCell(CONST, 1, DRIVER_VELOCITY_X, DRIVER_VELOCITY_Y);
        for (int y = 1; y < dimensions.y() - 1; ++y) {
            Coord<2> c1(0, y);
            Coord<2> c2(dimensions.x() - 1, y);

            if (box.inBounds(c1)) {
                grid->set(c1, driverCell);
            }
            if (box.inBounds(c2)) {
                grid->set(c2, driverCell);
            }
        }
    }

    bool inCircle(const Coord<2>& point,
                  const Coord<2>& center,
                  const int& diameter,
                  const double& xScale = 1.0)
    {
        Coord<2> delta = center - point;
        double dist = delta.x() * delta.x() * xScale +
            delta.y() * delta.y();
        return sqrt(dist) <= diameter;
    }

    void addWing(GridBase<Cell, 2> *grid)
    {
        CoordBox<2> box = grid->boundingBox();

        Coord<2> offset = Coord<2>(500, 950);
        // substract inherent offset
        offset -= Coord<2>(150, 100);

        // lower left forth circle
        for (int y = 100; y < 140; ++y) {
            for (int x = 150; x < 190; ++x) {
                Coord<2> c(x, y);
                c = c + offset;
                if (box.inBounds(c) &&
                    inCircle(c, Coord<2>(190, 140) + offset, 40)) {
                    grid->set(c, Cell(SOLID, 0));
                }
            }
        }

        // upper left forth circle
        for (int y = 140; y < 200; ++y) {
            for (int x = 150; x < 250; ++x) {
                Coord<2> c(x, y);
                c = c + offset;
                if (box.inBounds(c) &&
                    inCircle(c, Coord<2>(270, 140) + offset, 60, 0.25)) {
                    grid->set(c, Cell(SOLID, 0));
                }
            }
        }

        // left quadroid filler
        for (int y = 100; y < 140; ++y) {
            for (int x = 190; x < 250; ++x) {
                Coord<2> c(x, y);
                c = c + offset;
                if (box.inBounds(c)) {
                    grid->set(c, Cell(SOLID, 0));
                }
            }
        }

        // right circle fragment
        for (int y = 100; y < 200; ++y) {
            for (int x = 250; x < 350; ++x) {
                Coord<2> c(x, y);
                c = c + offset;
                if (box.inBounds(c) &&
                    inCircle(c, Coord<2>(250, -125) + offset, 325, 0.60)) {
                    grid->set(c, Cell(SOLID, 0));
                }
            }
        }

        // right triangle filler
        for (int x = 350; x < 600; ++x) {
            int maxY = 100 + 91.0 * (600 - x) / 250;
            for (int y = 100; y < maxY; ++y) {
                Coord<2> c(x, y);
                c = c + offset;
                if (box.inBounds(c)) {
                    grid->set(c, Cell(SOLID, 0));
                }
            }
        }
    }
};

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    Typemaps::initializeMaps();

    MPI::Aint displacements[] = {0};
    MPI::Datatype memberTypes[] = {MPI::CHAR};
    int lengths[] = {sizeof(Cell)};
    MPI::Datatype objType;
    objType =
        MPI::Datatype::Create_struct(1, lengths, displacements, memberTypes);
    objType.Commit();

    {
        AeroInitializer *init = new AeroInitializer(
            Coord<2>(MAX_X, MAX_Y),
            100000);

        StripingSimulator<Cell> sim(
            init,
            MPILayer().rank() ? 0 : new TracingBalancer(new NoOpBalancer()),
            1000,
            objType);

        sim.addWriter(
            new ParallelMPIIOWriter<Cell>(
                "snapshot",
                6000,
                init->maxSteps(),
                MPI::COMM_WORLD,
                objType));

        sim.addWriter(
            new BOVWriter<Cell, QuantitySelector>(
                "wing.quantity", 50));

        sim.addWriter(
            new BOVWriter<Cell, VelocitySelector>(
                "wing.velocity", 50));

        if (MPILayer().rank() == 0) {
            sim.addWriter(
                new TracingWriter<Cell>(
                    200, init->maxSteps()));
        }

        sim.run();
    }

    MPI_Finalize();
    return 0;
}
