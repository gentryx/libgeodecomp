/**
 * We need to include typemaps first to avoid problems with Intel
 * MPI's C++ bindings (which may collide with stdio.h's SEEK_SET,
 * SEEK_CUR etc.).
 */
#include <libgeodecomp/communication/typemaps.h>
#include <libgeodecomp/io/bovwriter.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/loadbalancer/noopbalancer.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>

using namespace LibGeoDecomp;

class Cell
{
public:
    static MPI_Datatype MPIDataType;

    class API :
        public APITraits::HasStencil<Stencils::Moore<3, 1> >,
        public APITraits::HasCubeTopology<3>,
        public APITraits::HasCustomMPIDataType<Cell>
    {};

    enum State {LIQUID, WEST_NOSLIP, EAST_NOSLIP, TOP, BOTTOM, NORTH_ACC, SOUTH_NOSLIP};

#define C 0
#define N 1
#define E 2
#define W 3
#define S 4
#define T 5
#define B 6

#define NW 7
#define SW 8
#define NE 9
#define SE 10

#define TW 11
#define BW 12
#define TE 13
#define BE 14

#define TN 15
#define BN 16
#define TS 17
#define BS 18

    inline explicit Cell(double v = 1.0, State s = LIQUID) :
        state(s)
    {
        comp[C] = v;
        for (int i = 1; i < 19; ++i) {
            comp[i] = 0.0;
        }
        density = 1.0;
    }

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, const unsigned& nanoStep)
    {
        *this = neighborhood[FixedCoord<0, 0>()];

        switch (state) {
        case LIQUID:
            updateFluid(neighborhood);
            break;
        case WEST_NOSLIP:
            updateWestNoSlip(neighborhood);
            break;
        case EAST_NOSLIP:
            updateEastNoSlip(neighborhood);
            break;
        case TOP :
            updateTop(neighborhood);
            break;
        case BOTTOM:
            updateBottom(neighborhood);
            break;
        case NORTH_ACC:
            updateNorthAcc(neighborhood);
            break;
        case SOUTH_NOSLIP:
            updateSouthNoSlip(neighborhood);
            break;
        }
    }

    template<typename COORD_MAP>
    void updateFluid(const COORD_MAP& neighborhood)
    {
#define GET_COMP(X, Y, Z, COMP) neighborhood[Coord<3>(X, Y, Z)].comp[COMP]
#define SQR(X) ((X)*(X))
        const double omega = 1.0/1.7;
        const double omega_trm = 1.0 - omega;
        const double omega_w0 = 3.0 * 1.0 / 3.0 * omega;
        const double omega_w1 = 3.0*1.0/18.0*omega;
        const double omega_w2 = 3.0*1.0/36.0*omega;
        const double one_third = 1.0 / 3.0;
        const int x = 0;
        const int y = 0;
        const int z = 0;
        double velX, velY, velZ;

        velX  =
            GET_COMP(x-1,y,z,E) + GET_COMP(x-1,y-1,z,NE) +
            GET_COMP(x-1,y+1,z,SE) + GET_COMP(x-1,y,z-1,TE) +
            GET_COMP(x-1,y,z+1,BE);
        velY  = GET_COMP(x,y-1,z,N) + GET_COMP(x+1,y-1,z,NW) +
            GET_COMP(x,y-1,z-1,TN) + GET_COMP(x,y-1,z+1,BN);
        velZ  = GET_COMP(x,y,z-1,T) + GET_COMP(x,y+1,z-1,TS) +
            GET_COMP(x+1,y,z-1,TW);

        const double rho =
            GET_COMP(x,y,z,C) + GET_COMP(x,y+1,z,S) +
            GET_COMP(x+1,y,z,W) + GET_COMP(x,y,z+1,B) +
            GET_COMP(x+1,y+1,z,SW) + GET_COMP(x,y+1,z+1,BS) +
            GET_COMP(x+1,y,z+1,BW) + velX + velY + velZ;
        velX  = velX
            - GET_COMP(x+1,y,z,W)    - GET_COMP(x+1,y-1,z,NW)
            - GET_COMP(x+1,y+1,z,SW) - GET_COMP(x+1,y,z-1,TW)
            - GET_COMP(x+1,y,z+1,BW);
        velY  = velY
            + GET_COMP(x-1,y-1,z,NE) - GET_COMP(x,y+1,z,S)
            - GET_COMP(x+1,y+1,z,SW) - GET_COMP(x-1,y+1,z,SE)
            - GET_COMP(x,y+1,z-1,TS) - GET_COMP(x,y+1,z+1,BS);
        velZ  = velZ+GET_COMP(x,y-1,z-1,TN) + GET_COMP(x-1,y,z-1,TE) - GET_COMP(x,y,z+1,B) - GET_COMP(x,y-1,z+1,BN) - GET_COMP(x,y+1,z+1,BS) - GET_COMP(x+1,y,z+1,BW) - GET_COMP(x-1,y,z+1,BE);

        density = rho;
        velocityX = velX;
        velocityX = velX;
        velocityY = velY;
        velocityZ = velZ;

        const double dir_indep_trm = one_third*rho - 0.5*( velX*velX + velY*velY + velZ*velZ );

        comp[C]=omega_trm * GET_COMP(x,y,z,C) + omega_w0*( dir_indep_trm );

        comp[NW]=omega_trm * GET_COMP(x+1,y-1,z,NW) +
            omega_w2*( dir_indep_trm - ( velX-velY ) + 1.5*SQR( velX-velY ) );
        comp[SE]=omega_trm * GET_COMP(x-1,y+1,z,SE) +
            omega_w2*( dir_indep_trm + ( velX-velY ) + 1.5*SQR( velX-velY ) );
        comp[NE]=omega_trm * GET_COMP(x-1,y-1,z,NE) +
            omega_w2*( dir_indep_trm + ( velX+velY ) + 1.5*SQR( velX+velY ) );
        comp[SW]=omega_trm * GET_COMP(x+1,y+1,z,SW) +
            omega_w2*( dir_indep_trm - ( velX+velY ) + 1.5*SQR( velX+velY ) );

        comp[TW]=omega_trm * GET_COMP(x+1,y,z-1,TW) + omega_w2*( dir_indep_trm - ( velX-velZ ) + 1.5*SQR( velX-velZ ) );
        comp[BE]=omega_trm * GET_COMP(x-1,y,z+1,BE) + omega_w2*( dir_indep_trm + ( velX-velZ ) + 1.5*SQR( velX-velZ ) );
        comp[TE]=omega_trm * GET_COMP(x-1,y,z-1,TE) + omega_w2*( dir_indep_trm + ( velX+velZ ) + 1.5*SQR( velX+velZ ) );
        comp[BW]=omega_trm * GET_COMP(x+1,y,z+1,BW) + omega_w2*( dir_indep_trm - ( velX+velZ ) + 1.5*SQR( velX+velZ ) );

        comp[TS]=omega_trm * GET_COMP(x,y+1,z-1,TS) + omega_w2*( dir_indep_trm - ( velY-velZ ) + 1.5*SQR( velY-velZ ) );
        comp[BN]=omega_trm * GET_COMP(x,y-1,z+1,BN) + omega_w2*( dir_indep_trm + ( velY-velZ ) + 1.5*SQR( velY-velZ ) );
        comp[TN]=omega_trm * GET_COMP(x,y-1,z-1,TN) + omega_w2*( dir_indep_trm + ( velY+velZ ) + 1.5*SQR( velY+velZ ) );
        comp[BS]=omega_trm * GET_COMP(x,y+1,z+1,BS) + omega_w2*( dir_indep_trm - ( velY+velZ ) + 1.5*SQR( velY+velZ ) );

        comp[N]=omega_trm * GET_COMP(x,y-1,z,N) + omega_w1*( dir_indep_trm + velY + 1.5*SQR(velY));
        comp[S]=omega_trm * GET_COMP(x,y+1,z,S) + omega_w1*( dir_indep_trm - velY + 1.5*SQR(velY));
        comp[E]=omega_trm * GET_COMP(x-1,y,z,E) + omega_w1*( dir_indep_trm + velX + 1.5*SQR(velX));
        comp[W]=omega_trm * GET_COMP(x+1,y,z,W) + omega_w1*( dir_indep_trm - velX + 1.5*SQR(velX));
        comp[T]=omega_trm * GET_COMP(x,y,z-1,T) + omega_w1*( dir_indep_trm + velZ + 1.5*SQR(velZ));
        comp[B]=omega_trm * GET_COMP(x,y,z+1,B) + omega_w1*( dir_indep_trm - velZ + 1.5*SQR(velZ));

    }

    template<typename COORD_MAP>
    void updateWestNoSlip(const COORD_MAP& neighborhood)
    {
        comp[E ]=GET_COMP(1, 0,  0, W);
        comp[NE]=GET_COMP(1, 1,  0, SW);
        comp[SE]=GET_COMP(1,-1,  0, NW);
        comp[TE]=GET_COMP(1, 0,  1, BW);
        comp[BE]=GET_COMP(1, 0, -1, TW);
    }

    template<typename COORD_MAP>
    void updateEastNoSlip(const COORD_MAP& neighborhood)
    {
        comp[W ]=GET_COMP(-1, 0, 0, E);
        comp[NW]=GET_COMP(-1, 0, 1, SE);
        comp[SW]=GET_COMP(-1,-1, 0, NE);
        comp[TW]=GET_COMP(-1, 0, 1, BE);
        comp[BW]=GET_COMP(-1, 0,-1, TE);
    }

    template<typename COORD_MAP>
    void updateTop(const COORD_MAP& neighborhood)
    {
        comp[B] =GET_COMP(0,0,-1,T);
        comp[BE]=GET_COMP(1,0,-1,TW);
        comp[BW]=GET_COMP(-1,0,-1,TE);
        comp[BN]=GET_COMP(0,1,-1,TS);
        comp[BS]=GET_COMP(0,-1,-1,TN);
    }

    template<typename COORD_MAP>
    void updateBottom(const COORD_MAP& neighborhood)
    {
        comp[T] =GET_COMP(0,0,1,B);
        comp[TE]=GET_COMP(1,0,1,BW);
        comp[TW]=GET_COMP(-1,0,1,BE);
        comp[TN]=GET_COMP(0,1,1,BS);
        comp[TS]=GET_COMP(0,-1,1,BN);
    }

    template<typename COORD_MAP>
    void updateNorthAcc(const COORD_MAP& neighborhood)
    {
        const double w_1 = 0.01;
        comp[S] =GET_COMP(0,-1,0,N);
        comp[SE]=GET_COMP(1,-1,0,NW)+6*w_1*0.1;
        comp[SW]=GET_COMP(-1,-1,0,NE)-6*w_1*0.1;
        comp[TS]=GET_COMP(0,-1,1,BN);
        comp[BS]=GET_COMP(0,-1,-1,TN);
    }

    template<typename COORD_MAP>
    void updateSouthNoSlip(const COORD_MAP& neighborhood)
    {
        comp[N] =GET_COMP(0,1,0,S);
        comp[NE]=GET_COMP(1,1,0,SW);
        comp[NW]=GET_COMP(-1,1,0,SE);
        comp[TN]=GET_COMP(0,1,1,BS);
        comp[BN]=GET_COMP(0,1,-1,TS);
    }

    double comp[19];
    double density;
    double velocityX;
    double velocityY;
    double velocityZ;
    State state;
};

MPI_Datatype Cell::MPIDataType;

class CellInitializer : public SimpleInitializer<Cell>
{
public:
    using SimpleInitializer<Cell>::gridDimensions;

    CellInitializer(Coord<3> dim, int maxSteps) : SimpleInitializer<Cell>(dim, maxSteps)
    {}

    virtual void grid(GridBase<Cell, 3> *ret)
    {
        CoordBox<3> box = ret->boundingBox();
        Coord<3> size = gridDimensions();

        for (int z = 0; z < size.z(); ++z) {
            for (int y = 0; y < size.y(); ++y) {
                for (int x = 0; x < size.x(); ++x) {
                    Coord<3> c(x, y, z);
                    Cell::State s = Cell::LIQUID;

                    if (c.x() == 0) {
                        s = Cell::WEST_NOSLIP;
                    }
                    if (c.x() == (size.x() - 1)) {
                        s = Cell::EAST_NOSLIP;
                    }

                    if (c.y() == 0) {
                        s = Cell::SOUTH_NOSLIP;
                    }
                    if (c.y() == (size.y() - 1)) {
                        s = Cell::NORTH_ACC;
                    }

                    if (c.z() == 0) {
                        s = Cell::BOTTOM;
                    }
                    if (c.z() == (size.z() - 1)) {
                        s = Cell::TOP;
                    }

                    if (box.inBounds(c)) {
                        ret->set(c, Cell(0, s));
                    }
                }
            }
        }
    }
};

class DensitySelector
{
public:
    typedef double VariableType;

    static std::string varName()
    {
        return "density";
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
        *storage = cell.density;
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
        storage[2] = cell.velocityZ;
    }
};

void runSimulation()
{
    MPI_Aint displacements[] = { 0 };
    MPI_Datatype memberTypes[] = { MPI_CHAR };
    int lengths[] = { sizeof(Cell) };
    MPI_Type_create_struct(1, lengths, displacements, memberTypes, &Cell::MPIDataType);
    MPI_Type_commit(&Cell::MPIDataType);

    int outputFrequency = 1000;
    CellInitializer *init = new CellInitializer(Coord<3>(200, 200, 200), 2000000);

    StripingSimulator<Cell> sim(
        init,
        MPILayer().rank() ? 0 : new TracingBalancer(new NoOpBalancer()),
        1000000);

    sim.addWriter(
        new BOVWriter<Cell>(Selector<Cell>(&Cell::density,   "density"),
                            "lbm.density",
                            outputFrequency));
    sim.addWriter(
        new BOVWriter<Cell>(Selector<Cell>(&Cell::velocityX, "velocityX"),
                            "lbm.velocityX",
                            outputFrequency));
    sim.addWriter(
        new BOVWriter<Cell>(Selector<Cell>(&Cell::velocityY, "velocityY"),
                            "lbm.velocityY",
                            outputFrequency));
    sim.addWriter(
        new BOVWriter<Cell>(Selector<Cell>(&Cell::velocityZ, "velocityZ"),
                            "lbm.velocityZ",
                            outputFrequency));

    if (MPILayer().rank() == 0) {
        sim.addWriter(
            new TracingWriter<Cell>(
                20,
                init->maxSteps()));
    }

    sim.run();
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    runSimulation();

    MPI_Finalize();
    return 0;
}
