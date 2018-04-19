#include <libgeodecomp/config.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/cpubenchmark.h>
#include <libgeodecomp/parallelization/openmpsimulator.h>
#include <libflatarray/flat_array.hpp>

#include "lbmsoa.h"

using namespace LibGeoDecomp;

// Padding is fine. Also, don't warn about inlining.
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4710 4711 4820 )
#endif

class LBMSoACell
{
public:
    typedef LibFlatArray::short_vec<double, 16> Double;

    class API : public APITraits::HasFixedCoordsOnlyUpdate,
                public APITraits::HasSoA,
                public APITraits::HasUpdateLineX,
                public APITraits::HasStencil<Stencils::Moore<3, 1> >,
                public APITraits::HasCubeTopology<3>
    {};

    enum State {LIQUID, WEST_NOSLIP, EAST_NOSLIP, TOP, BOTTOM, NORTH_ACC, SOUTH_NOSLIP};

    inline explicit LBMSoACell(double v=1.0, const State& s=LIQUID) :
        C(v),
        N(0),
        E(0),
        W(0),
        S(0),
        T(0),
        B(0),

        NW(0),
        SW(0),
        NE(0),
        SE(0),

        TW(0),
        BW(0),
        TE(0),
        BE(0),

        TN(0),
        BN(0),
        TS(0),
        BS(0),

        density(1.0),
        velocityX(0),
        velocityY(0),
        velocityZ(0),
        state(s)
    {}

    template<typename ACCESSOR1, typename ACCESSOR2>
    static void updateLineX(
        ACCESSOR1& hoodOld,
        int indexEnd,
        ACCESSOR2& hoodNew,
        unsigned /* nanoStep */)
    {
        updateLineXFluid(hoodOld, indexEnd, hoodNew);
    }

    // ignore constant overflows here as they're caused by compiling
    // for very large grids on a 32-bit machine:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4307 )
#endif

    template<typename ACCESSOR1, typename ACCESSOR2>
    static void updateLineXFluid(
        ACCESSOR1& hoodOld,
        int indexEnd,
        ACCESSOR2& hoodNew)
    {
#define GET_COMP(X, Y, Z, COMP) Double(&hoodOld[FixedCoord<X, Y, Z>()].COMP())
#define SQR(X) ((X)*(X))
        const Double omega = 1.0/1.7;
        const Double omega_trm = Double(1.0) - omega;
        const Double omega_w0 = Double(3.0 * 1.0 / 3.0) * omega;
        const Double omega_w1 = Double(3.0*1.0/18.0)*omega;
        const Double omega_w2 = Double(3.0*1.0/36.0)*omega;
        const Double one_third = 1.0 / 3.0;
        const Double one_half = 0.5;
        const Double one_point_five = 1.5;

        const int x = 0;
        const int y = 0;
        const int z = 0;
        Double velX, velY, velZ;

        for (; hoodOld.index() < indexEnd; hoodNew.index() += Double::ARITY, hoodOld.index() += Double::ARITY) {
            velX  =
                GET_COMP(x-1,y,z,E) + GET_COMP(x-1,y-1,z,NE) +
                GET_COMP(x-1,y+1,z,SE) + GET_COMP(x-1,y,z-1,TE) +
                GET_COMP(x-1,y,z+1,BE);
            velY  =
                GET_COMP(x,y-1,z,N) + GET_COMP(x+1,y-1,z,NW) +
                GET_COMP(x,y-1,z-1,TN) + GET_COMP(x,y-1,z+1,BN);
            velZ  =
                GET_COMP(x,y,z-1,T) + GET_COMP(x,y+1,z-1,TS) +
                GET_COMP(x+1,y,z-1,TW);

            const Double rho =
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

            &hoodNew.density() << rho;
            &hoodNew.velocityX() << velX;
            &hoodNew.velocityY() << velY;
            &hoodNew.velocityZ() << velZ;

            const Double dir_indep_trm = one_third*rho - one_half *( velX*velX + velY*velY + velZ*velZ );

            &hoodNew.C() << (omega_trm * GET_COMP(x,y,z,C) + omega_w0 * dir_indep_trm );

            &hoodNew.NW() << (omega_trm * GET_COMP(x+1,y-1,z,NW) + omega_w2*( dir_indep_trm - ( velX - velY ) + one_point_five * SQR( velX-velY ) ));
            &hoodNew.SE() << (omega_trm * GET_COMP(x-1,y+1,z,SE) + omega_w2*( dir_indep_trm + ( velX - velY ) + one_point_five * SQR( velX-velY ) ));
            &hoodNew.NE() << (omega_trm * GET_COMP(x-1,y-1,z,NE) + omega_w2*( dir_indep_trm + ( velX + velY ) + one_point_five * SQR( velX+velY ) ));
            &hoodNew.SW() << (omega_trm * GET_COMP(x+1,y+1,z,SW) + omega_w2*( dir_indep_trm - ( velX + velY ) + one_point_five * SQR( velX+velY ) ));

            &hoodNew.TW() << (omega_trm * GET_COMP(x+1,y,z-1,TW) + omega_w2*( dir_indep_trm - ( velX - velZ ) + one_point_five * SQR( velX-velZ ) ));
            &hoodNew.BE() << (omega_trm * GET_COMP(x-1,y,z+1,BE) + omega_w2*( dir_indep_trm + ( velX - velZ ) + one_point_five * SQR( velX-velZ ) ));
            &hoodNew.TE() << (omega_trm * GET_COMP(x-1,y,z-1,TE) + omega_w2*( dir_indep_trm + ( velX + velZ ) + one_point_five * SQR( velX+velZ ) ));
            &hoodNew.BW() << (omega_trm * GET_COMP(x+1,y,z+1,BW) + omega_w2*( dir_indep_trm - ( velX + velZ ) + one_point_five * SQR( velX+velZ ) ));

            &hoodNew.TS() << (omega_trm * GET_COMP(x,y+1,z-1,TS) + omega_w2*( dir_indep_trm - ( velY - velZ ) + one_point_five * SQR( velY-velZ ) ));
            &hoodNew.BN() << (omega_trm * GET_COMP(x,y-1,z+1,BN) + omega_w2*( dir_indep_trm + ( velY - velZ ) + one_point_five * SQR( velY-velZ ) ));
            &hoodNew.TN() << (omega_trm * GET_COMP(x,y-1,z-1,TN) + omega_w2*( dir_indep_trm + ( velY + velZ ) + one_point_five * SQR( velY+velZ ) ));
            &hoodNew.BS() << (omega_trm * GET_COMP(x,y+1,z+1,BS) + omega_w2*( dir_indep_trm - ( velY + velZ ) + one_point_five * SQR( velY+velZ ) ));

            &hoodNew.N() << (omega_trm * GET_COMP(x,y-1,z,N) + omega_w1*( dir_indep_trm + velY + one_point_five * SQR(velY)));
            &hoodNew.S() << (omega_trm * GET_COMP(x,y+1,z,S) + omega_w1*( dir_indep_trm - velY + one_point_five * SQR(velY)));
            &hoodNew.E() << (omega_trm * GET_COMP(x-1,y,z,E) + omega_w1*( dir_indep_trm + velX + one_point_five * SQR(velX)));
            &hoodNew.W() << (omega_trm * GET_COMP(x+1,y,z,W) + omega_w1*( dir_indep_trm - velX + one_point_five * SQR(velX)));
            &hoodNew.T() << (omega_trm * GET_COMP(x,y,z-1,T) + omega_w1*( dir_indep_trm + velZ + one_point_five * SQR(velZ)));
            &hoodNew.B() << (omega_trm * GET_COMP(x,y,z+1,B) + omega_w1*( dir_indep_trm - velZ + one_point_five * SQR(velZ)));
        }
    }

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

    double C;
    double N;
    double E;
    double W;
    double S;
    double T;
    double B;

    double NW;
    double SW;
    double NE;
    double SE;

    double TW;
    double BW;
    double TE;
    double BE;

    double TN;
    double BN;
    double TS;
    double BS;

    double density;
    double velocityX;
    double velocityY;
    double velocityZ;
    State state;

};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

// Removing inline functions are OK. Also, ignore constant overflows
// here as they're caused by compiling for very large grids on a
// 32-bit machine:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4307 4514 4710 4711 )
#endif

LIBFLATARRAY_REGISTER_SOA(
    LBMSoACell,
    ((double)(C))
    ((double)(N))
    ((double)(E))
    ((double)(W))
    ((double)(S))
    ((double)(T))
    ((double)(B))
    ((double)(NW))
    ((double)(SW))
    ((double)(NE))
    ((double)(SE))
    ((double)(TW))
    ((double)(BW))
    ((double)(TE))
    ((double)(BE))
    ((double)(TN))
    ((double)(BN))
    ((double)(TS))
    ((double)(BS))
    ((double)(density))
    ((double)(velocityX))
    ((double)(velocityY))
    ((double)(velocityZ))
    ((LBMSoACell::State)(state))
                          )
#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

std::string LBMSoA::family()
{
    return "LBM";
}

std::string LBMSoA::species()
{
    return "gold";
}

double LBMSoA::performance(std::vector<int> rawDim)
{
    Coord<3> dim(rawDim[0], rawDim[1], rawDim[2]);
    unsigned maxT = 200;
    OpenMPSimulator<LBMSoACell> sim(
        new NoOpInitializer<LBMSoACell>(dim, maxT));

    double seconds = 0;
    {
        ScopedTimer t(&seconds);

        sim.run();
    }

    if (sim.getGrid()->get(Coord<3>(1, 1, 1)).density == 4711) {
        std::cout << "this statement just serves to prevent the compiler from"
                  << "optimizing away the loops above\n";
    }

    double updates = 1.0 * maxT * dim.prod();
    double gLUPS = 1e-9 * updates / seconds;

    return gLUPS;
}

std::string LBMSoA::unit()
{
    return "GLUPS";
}

LIBFLATARRAY_DISABLE_SYSTEM_HEADER_WARNINGS_EOF
