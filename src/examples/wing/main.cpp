#include <cmath>
/**
 * We need to include typemaps first to avoid problems with Intel
 * MPI's C++ bindings (which may collide with stdio.h's SEEK_SET,
 * SEEK_CUR etc.).
 */
#include <libgeodecomp/communication/typemaps.h>
#include <libgeodecomp/io/bovwriter.h>
#include <libgeodecomp/io/parallelmpiiowriter.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/loadbalancer/noopbalancer.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>

using namespace LibGeoDecomp;

const double dX = 1.0;
const double dT = 1.0;

const double FLOW_DIFFUSION = 0.1 * dT * dX;
const double PRESSURE_DIFFUSION = 0.1 * dT * dX;

const double PRESSURE_SPEED = 0.1;

const double FRICTION = 0.00015;
const double DRIVER_VELOCITY_X = 3.0;
const double DRIVER_VELOCITY_Y = 0.0;

enum State {LIQUID=0, SLIP=1, SOLID=2, CONST=3};

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

double INFLUENCES[] =
    {INFLUENCE_FACTOR * FLOW_DIFFUSION, FLOW_DIFFUSION, INFLUENCE_FACTOR * FLOW_DIFFUSION,
     FLOW_DIFFUSION, FLOW_DIFFUSION,
     INFLUENCE_FACTOR * FLOW_DIFFUSION, FLOW_DIFFUSION, INFLUENCE_FACTOR * FLOW_DIFFUSION};

double LENGTHS[] =
    {DIAG, 1.0, DIAG,
     1.0, 1.0,
     DIAG, 1.0, DIAG};

template<int I>
class NEIGHBORS;

template<>
class NEIGHBORS<0> : public FixedCoord<-1, -1>
{};

template<>
class NEIGHBORS<1> : public FixedCoord< 0, -1>
{};

template<>
class NEIGHBORS<2> : public FixedCoord< 1, -1>
{};

template<>
class NEIGHBORS<3> : public FixedCoord<-1,  0>
{};

template<>
class NEIGHBORS<4> : public FixedCoord< 1,  0>
{};

template<>
class NEIGHBORS<5> : public FixedCoord<-1,  1>
{};

template<>
class NEIGHBORS<6> : public FixedCoord< 0,  1>
{};

template<>
class NEIGHBORS<7> : public FixedCoord< 1,  1>
{};


class Cell
{
public:
    friend int main(int argc, char **argv);
    static MPI_Datatype MPIDataType;

    class API :
        public APITraits::HasCustomMPIDataType<Cell>,
        public APITraits::HasFixedCoordsOnlyUpdate
    {};

    explicit Cell(
        State state = LIQUID,
        double quantity = 0,
        double velocityX = 0,
        double velocityY = 0) :
        state(state),
        quantity(quantity),
        velocityX(velocityX),
        velocityY(velocityY)
    {}

    template<class NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& neighbors, const int& nanoStep)
    {
        *this = neighbors[FixedCoord<0, 0>()];

        if (state == SOLID) {
            return;
        }

        diffuse(neighbors);
    }

private:
    template<int I>
    static void flux(const Cell& from, const Cell& to,
                     double *fluxFlow, double *fluxPressure)
    {
        if (from.state == SOLID || to.state == SOLID) {
            *fluxFlow = 0;
            *fluxPressure = 0;
            return;
        }

        typedef NEIGHBORS<I> DIR;
        const double& influence = INFLUENCES[I];

        *fluxFlow = (DIR::X * from.velocityX + DIR::Y * from.velocityY) *
            influence * from.quantity;
        *fluxFlow = std::max(0.0, *fluxFlow);

        *fluxPressure = 0;
        if (from.quantity > to.quantity) {
            *fluxPressure = (from.quantity - to.quantity) * PRESSURE_DIFFUSION;
        }
    }

    template<int I, class NEIGHBORHOOD>
    static void addFlowFromNeighbor(
        const Cell& oldSelf,
        const NEIGHBORHOOD& neighbors,
        double *fluxVelocityX,
        double *fluxVelocityY,
        double *newQuantity)
    {
        typedef NEIGHBORS<I> DIR;
        const Cell& other = neighbors[DIR()];
        const double& length = LENGTHS[I];

        double fluxFlow;
        double fluxPressure;
        flux<7 - I>(other, oldSelf, &fluxFlow, &fluxPressure);

        if (fluxFlow == 0 && fluxPressure == 0) {
            return;
        }

        double totalFlow = fluxFlow + fluxPressure;
        *fluxVelocityX += totalFlow * other.velocityX;
        *fluxVelocityY += totalFlow * other.velocityY;
        *newQuantity += totalFlow;

        double pressureCoefficient = length * PRESSURE_SPEED * fluxPressure;
        *fluxVelocityX += -DIR::X * pressureCoefficient;
        *fluxVelocityY += -DIR::Y * pressureCoefficient;
    }

    template<int I, class NEIGHBORHOOD>
    static void removeFlowToNeighbor(
        const Cell& oldSelf,
        const NEIGHBORHOOD& neighbors,
        double *newQuantity)
    {
        typedef NEIGHBORS<I> DIR;
        const Cell& other = neighbors[DIR()];
        double fluxFlow;
        double fluxPressure;
        flux<I>(oldSelf, other, &fluxFlow, &fluxPressure);

        *newQuantity -= fluxFlow;
        *newQuantity -= fluxPressure;
    }

#define CYCLE_NEIGHBORS_3(FUNCTION, P1, P2, P3) \
    FUNCTION<0>(P1, P2, P3);                    \
    FUNCTION<1>(P1, P2, P3);                    \
    FUNCTION<2>(P1, P2, P3);                    \
    FUNCTION<3>(P1, P2, P3);                    \
    FUNCTION<4>(P1, P2, P3);                    \
    FUNCTION<5>(P1, P2, P3);                    \
    FUNCTION<6>(P1, P2, P3);                    \
    FUNCTION<7>(P1, P2, P3);

#define CYCLE_NEIGHBORS_5(FUNCTION, P1, P2, P3, P4, P5) \
    FUNCTION<0>(P1, P2, P3, P4, P5);                    \
    FUNCTION<1>(P1, P2, P3, P4, P5);                    \
    FUNCTION<2>(P1, P2, P3, P4, P5);                    \
    FUNCTION<3>(P1, P2, P3, P4, P5);                    \
    FUNCTION<4>(P1, P2, P3, P4, P5);                    \
    FUNCTION<5>(P1, P2, P3, P4, P5);                    \
    FUNCTION<6>(P1, P2, P3, P4, P5);                    \
    FUNCTION<7>(P1, P2, P3, P4, P5);

    template<class NEIGHBORHOOD>
    void diffuse(const NEIGHBORHOOD& neighbors)
    {
        double fluxVelocityX = 0;
        double fluxVelocityY = 0;
        double newQuantity = quantity;
        const Cell& oldSelf = neighbors[FixedCoord<0, 0>()];

        CYCLE_NEIGHBORS_5(
            addFlowFromNeighbor, oldSelf, neighbors, &fluxVelocityX, &fluxVelocityY, &newQuantity);

        double velocityCoeff = (state == SLIP) ? (1 - FRICTION) : 1.0;
        velocityCoeff /= newQuantity;
        if (state != CONST) {
            velocityX = (quantity * velocityX + fluxVelocityX) * velocityCoeff;
            velocityY = (quantity * velocityY + fluxVelocityY) * velocityCoeff;
        }

        CYCLE_NEIGHBORS_3(
            removeFlowToNeighbor, oldSelf, neighbors, &newQuantity);

        if (newQuantity < 0) {
            std::cerr << "ohoh\n"
                      << "oldSelf:\n"
                      << "  quantity = " << oldSelf.quantity << "\n"
                      << "  velocityX = " << oldSelf.velocityX << "\n"
                      << "  velocityY = " << oldSelf.velocityY << "\n"
                      << "newSelf:\n"
                      << "  quantity = " << newQuantity << "\n"
                      << "  velocityX = " << velocityX << "\n"
                      << "  velocityY = " << velocityY << "\n\n";

            throw std::logic_error("negative quantity, unstable simulation!");
        }

        quantity = newQuantity;
    }

    State state;
    double quantity;
    double velocityX;
    double velocityY;
};

MPI_Datatype Cell::MPIDataType;

class AeroInitializer : public LibGeoDecomp::SimpleInitializer<Cell>
{
public:
    using LibGeoDecomp::SimpleInitializer<Cell>::dimensions;

    class Setup
    {
    public:
        virtual ~Setup()
        {}

        virtual Coord<2> getMax() const = 0;
        virtual void addCells(GridBase<Cell, 2> *grid) = 0;

        Coord<2> dimensions() const
        {
            return getMax();
        }


    };

    class WingSetup : public Setup
    {
    public:
        Coord<2> getMax() const
        {
            return Coord<2>(2048, 2048);
        }

        void addCells(GridBase<Cell, 2> *grid)
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

    private:
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
    };

    class WingWithInletSetup : public WingSetup
    {
    public:
        void addCells(GridBase<Cell, 2> *grid)
        {
            WingSetup::addCells(grid);

            CoordBox<2> box = grid->boundingBox();

            Cell driverCell(CONST, 1, DRIVER_VELOCITY_X, DRIVER_VELOCITY_Y);
            for (int y = 1; y < dimensions().y() - 1; ++y) {
                Coord<2> c1(0, y);
                Coord<2> c2(dimensions().x() - 1, y);

                if (box.inBounds(c1)) {
                    grid->set(c1, driverCell);
                }
                if (box.inBounds(c2)) {
                    grid->set(c2, driverCell);
                }
            }
        }
    };

    class LidSetup : public Setup
    {
    public:
        Coord<2> getMax() const
        {
            return Coord<2>(512, 512);
        }

        void addCells(GridBase<Cell, 2> *grid)
        {
            CoordBox<2> box = grid->boundingBox();

            Cell driverCell(CONST, 1, DRIVER_VELOCITY_X, DRIVER_VELOCITY_Y);
            Cell slipCell(SLIP, 1);

            for (int y = 0; y < dimensions().y(); ++y) {
                Coord<2> c1(0, y);
                Coord<2> c2(dimensions().x() - 1, y);
                if (box.inBounds(c1)) {
                    grid->set(c1, slipCell);
                }
                if (box.inBounds(c2)) {
                    grid->set(c2, slipCell);
                }
            }

            for (int x = 0; x < dimensions().x(); ++x) {
                Coord<2> c(x, 0);
                if (box.inBounds(c)) {
                    grid->set(c, slipCell);
                }
            }

            for (int x = 1; x < dimensions().x() - 1; ++x) {
                Coord<2> c(x, dimensions().y() - 1);
                if (box.inBounds(c)) {
                    grid->set(c, driverCell);
                }
            }
        }
    };

    AeroInitializer(
        Setup *setup,
        std::size_t steps) :
        SimpleInitializer<Cell>(setup->getMax(), steps),
        setup(setup)
    {}

    virtual void grid(GridBase<Cell, 2> *grid)
    {
        CoordBox<2> box = grid->boundingBox();
        grid->setEdge(Cell(SOLID));

        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            grid->set(*i, Cell(LIQUID, 1));
        }

        setup->addCells(grid);
    }

private:
    boost::shared_ptr<Setup> setup;
};

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    MPI_Aint displacements[] = { 0 };
    MPI_Datatype memberTypes[] = { MPI_CHAR };
    int lengths[] = { sizeof(Cell) };
    MPI_Type_create_struct(1, lengths, displacements, memberTypes, &Cell::MPIDataType);
    MPI_Type_commit(&Cell::MPIDataType);

    if (argc != 2) {
        std::cerr << "USAGE: " << argv[0] << "  PRESET" << std::endl
                  << "  with PRESET in {WING, WING_WITH_INLET, LID}" << std::endl;
        MPI_Finalize();
        return 1;
    }

    AeroInitializer::Setup *setup = 0;
    if (argv[1] == std::string("WING")) {
        setup = new AeroInitializer::WingSetup;
    }
    if (argv[1] == std::string("WING_WITH_INLET")) {
        setup = new AeroInitializer::WingWithInletSetup;
    }
    if (argv[1] == std::string("LID")) {
        setup = new AeroInitializer::LidSetup;
    }

    if (setup == 0) {
        std::cerr << "ERROR: unknown preset" << std::endl;
        MPI_Finalize();
        return 2;
    }

    {
        AeroInitializer *init = new AeroInitializer(
            setup,
            200000);

        StripingSimulator<Cell> sim(
            init,
            MPILayer().rank() ? 0 : new TracingBalancer(new NoOpBalancer()),
            1000);

        sim.addWriter(
            new ParallelMPIIOWriter<Cell>(
                "snapshot",
                6000,
                init->maxSteps(),
                MPI_COMM_WORLD));


        sim.addWriter(
            new BOVWriter<Cell>(
                Selector<Cell>(&Cell::quantity,  "quantity"),  "wing.quantity",  100));

        sim.addWriter(
            new BOVWriter<Cell>(
                Selector<Cell>(&Cell::velocityX, "velocityX"), "wing.velocityX", 100));

        sim.addWriter(
            new BOVWriter<Cell>(
                Selector<Cell>(&Cell::velocityY, "velocityY"), "wing.velocityY", 100));

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
