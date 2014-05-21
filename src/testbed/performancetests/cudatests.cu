#include <cuda.h>
#include <iostream>
#include <stdexcept>

#include <libflatarray/testbed/gpu_benchmark.hpp>
#include <libflatarray/testbed/evaluate.hpp>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/misc/cudautil.h>
#include <libgeodecomp/storage/fixedneighborhood.h>
#include <libgeodecomp/storage/soagrid.h>

using namespace LibGeoDecomp;

class GPUBenchmark : public LibFlatArray::gpu_benchmark
{
public:
    double performance(std::vector<int> dim)
    {
        Coord<3> c(dim[0], dim[1], dim[2]);
        return performance2(c);
    }

    virtual double performance2(const Coord<3>& dim) = 0;
};

class Cell
{
public:
    double c;
    int a;
    char b;
};

LIBFLATARRAY_REGISTER_SOA(Cell, ((double)(c))((int)(a))((char)(b)))

class CellLBM
{
public:
    class API :
        public APITraits::HasStencil<Stencils::Moore<3, 1> >,
        public APITraits::HasCubeTopology<3>
    {};

    double C;
    double N;
    double E;
    double W;
    double S;
    double T;
    double B;
    double NW;
    double NE;
    double SW;
    double SE;
    double TW;
    double BW;
    double TE;
    double BE;
    double TN;
    double BN;
    double TS;
    double BS;
};

LIBFLATARRAY_REGISTER_SOA(CellLBM, ((double)(C))((double)(N))((double)(E))((double)(W))((double)(S))((double)(T))((double)(B))((double)(NW))((double)(SW))((double)(NE))((double)(SE))((double)(TW))((double)(BW))((double)(TE))((double)(BE))((double)(TN))((double)(BN))((double)(TS))((double)(BS)))

#define hoody(X, Y, Z)                                                  \
    gridOld[z * dimX * dimY + y * dimX + x + X + Y * dimX + Z * dimX * dimY]

template<int DIM_X, int DIM_Y, int DIM_Z>
__global__ void updateRTMClassic(int dimX, int dimY, int dimZ, double *gridOld, double *gridNew)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x + 2;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 2;
    int z = 2;

    double c0 = hoody(0, 0, -2);
    double c1 = hoody(0, 0, -1);
    double c2 = hoody(0, 0,  0);
    double c3 = hoody(0, 0,  1);

#pragma unroll 10
    for (; z < (dimZ - 2); ++z) {
        double c4 = hoody(0, 0, 2);

        gridNew[z * dimX * dimY + y * dimX + x] =
            0.10 * c0 +
            0.15 * c1 +
            0.20 * c2 +
            0.25 * c4 +
            0.30 * hoody( 0, -2, 0) +
            0.35 * hoody( 0, -1, 0) +
            0.40 * hoody( 0,  1, 0) +
            0.45 * hoody( 0,  2, 0) +
            0.50 * hoody(-2,  0, 0) +
            0.55 * hoody(-1,  0, 0) +
            0.60 * hoody( 1,  0, 0) +
            0.65 * hoody( 2,  0, 0);

        c0 = c1;
        c1 = c2;
        c2 = c3;
        c3 = c4;
    }
}

#undef hoody

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

#define GET_COMP(X, Y, Z, DIR)                                          \
    gridOld[(Z) * dimX * dimY + (Y) * dimX + (X) + (DIR) * dimX * dimY * dimZ]

#define SET_COMP(DIR)                                                   \
    gridNew[z   * dimX * dimY +   y * dimX +   x + (DIR) * dimX * dimY * dimZ]

template<int DIM_X, int DIM_Y, int DIM_Z>
__global__ void updateLBMClassic(int dimX, int dimY, int dimZ, double *gridOld, double *gridNew)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x + 2;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 2;
    int z = 2;

#pragma unroll 10
    for (; z < (dimZ - 2); z += 1) {

#define SQR(X) ((X)*(X))
        const double omega = 1.0/1.7;
        const double omega_trm = 1.0 - omega;
        const double omega_w0 = 3.0 * 1.0 / 3.0 * omega;
        const double omega_w1 = 3.0*1.0/18.0*omega;
        const double omega_w2 = 3.0*1.0/36.0*omega;
        const double one_third = 1.0 / 3.0;
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

        // density = rho;
        // velocityX = velX;
        // velocityY = velY;
        // velocityZ = velZ;

        const double dir_indep_trm = one_third*rho - 0.5*( velX*velX + velY*velY + velZ*velZ );

        SET_COMP(C)=omega_trm * GET_COMP(x,y,z,C) + omega_w0*( dir_indep_trm );

        SET_COMP(NW)=omega_trm * GET_COMP(x+1,y-1,z,NW) +
            omega_w2*( dir_indep_trm - ( velX-velY ) + 1.5*SQR( velX-velY ) );
        SET_COMP(SE)=omega_trm * GET_COMP(x-1,y+1,z,SE) +
            omega_w2*( dir_indep_trm + ( velX-velY ) + 1.5*SQR( velX-velY ) );
        SET_COMP(NE)=omega_trm * GET_COMP(x-1,y-1,z,NE) +
            omega_w2*( dir_indep_trm + ( velX+velY ) + 1.5*SQR( velX+velY ) );
        SET_COMP(SW)=omega_trm * GET_COMP(x+1,y+1,z,SW) +
            omega_w2*( dir_indep_trm - ( velX+velY ) + 1.5*SQR( velX+velY ) );

        SET_COMP(TW)=omega_trm * GET_COMP(x+1,y,z-1,TW) + omega_w2*( dir_indep_trm - ( velX-velZ ) + 1.5*SQR( velX-velZ ) );
        SET_COMP(BE)=omega_trm * GET_COMP(x-1,y,z+1,BE) + omega_w2*( dir_indep_trm + ( velX-velZ ) + 1.5*SQR( velX-velZ ) );
        SET_COMP(TE)=omega_trm * GET_COMP(x-1,y,z-1,TE) + omega_w2*( dir_indep_trm + ( velX+velZ ) + 1.5*SQR( velX+velZ ) );
        SET_COMP(BW)=omega_trm * GET_COMP(x+1,y,z+1,BW) + omega_w2*( dir_indep_trm - ( velX+velZ ) + 1.5*SQR( velX+velZ ) );

        SET_COMP(TS)=omega_trm * GET_COMP(x,y+1,z-1,TS) + omega_w2*( dir_indep_trm - ( velY-velZ ) + 1.5*SQR( velY-velZ ) );
        SET_COMP(BN)=omega_trm * GET_COMP(x,y-1,z+1,BN) + omega_w2*( dir_indep_trm + ( velY-velZ ) + 1.5*SQR( velY-velZ ) );
        SET_COMP(TN)=omega_trm * GET_COMP(x,y-1,z-1,TN) + omega_w2*( dir_indep_trm + ( velY+velZ ) + 1.5*SQR( velY+velZ ) );
        SET_COMP(BS)=omega_trm * GET_COMP(x,y+1,z+1,BS) + omega_w2*( dir_indep_trm - ( velY+velZ ) + 1.5*SQR( velY+velZ ) );

        SET_COMP(N)=omega_trm * GET_COMP(x,y-1,z,N) + omega_w1*( dir_indep_trm + velY + 1.5*SQR(velY));
        SET_COMP(S)=omega_trm * GET_COMP(x,y+1,z,S) + omega_w1*( dir_indep_trm - velY + 1.5*SQR(velY));
        SET_COMP(E)=omega_trm * GET_COMP(x-1,y,z,E) + omega_w1*( dir_indep_trm + velX + 1.5*SQR(velX));
        SET_COMP(W)=omega_trm * GET_COMP(x+1,y,z,W) + omega_w1*( dir_indep_trm - velX + 1.5*SQR(velX));
        SET_COMP(T)=omega_trm * GET_COMP(x,y,z-1,T) + omega_w1*( dir_indep_trm + velZ + 1.5*SQR(velZ));
        SET_COMP(B)=omega_trm * GET_COMP(x,y,z+1,B) + omega_w1*( dir_indep_trm - velZ + 1.5*SQR(velZ));
    }
}

#undef GET_COMP
#undef SET_COMP

#undef C
#undef N
#undef E
#undef W
#undef S
#undef T
#undef B

#undef NW
#undef SW
#undef NE
#undef SE

#undef TW
#undef BW
#undef TE
#undef BE

#undef TN
#undef BN
#undef TS
#undef BS

#define hoody(X, Y, Z)                          \
    hoodOld[LibFlatArray::coord<X, Y, Z>()]

template<int DIM_X, int DIM_Y, int DIM_Z>
__global__ void updateRTMSoA(int dimX, int dimY, int dimZ, double *gridOld, double *gridNew)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x + 2;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 2;
    int z = 2;

    int index = z * DIM_X * DIM_Y + y * DIM_X + x;
    int offset = DIM_X * DIM_Y;
    int end = DIM_X * DIM_Y * (dimZ - 2);

    LibFlatArray::soa_accessor_light<Cell, DIM_X, DIM_Y, DIM_Z, 0> hoodNew((char*)gridNew, index);
    LibFlatArray::soa_accessor_light<Cell, DIM_X, DIM_Y, DIM_Z, 0> hoodOld((char*)gridOld, index);

    double c0 = hoody(0, 0, -2).c();
    double c1 = hoody(0, 0, -1).c();
    double c2 = hoody(0, 0,  0).c();
    double c3 = hoody(0, 0,  1).c();

#pragma unroll 10
    for (; index < end; index += offset) {
        double c4 = hoody(0, 0, 2).c();
        hoodNew[LibFlatArray::coord<0, 0, 0>()].c() =
            0.10 * c0 +
            0.15 * c1 +
            0.20 * c2 +
            0.25 * c4 +
            0.30 * hoody( 0, -2, 0).c() +
            0.35 * hoody( 0, -1, 0).c() +
            0.40 * hoody( 0,  1, 0).c() +
            0.45 * hoody( 0,  2, 0).c() +
            0.50 * hoody(-2,  0, 0).c() +
            0.55 * hoody(-1,  0, 0).c() +
            0.60 * hoody( 1,  0, 0).c() +
            0.65 * hoody( 2,  0, 0).c();

        c0 = c1;
        c1 = c2;
        c2 = c3;
        c3 = c4;
    }
}

#undef hoody

#define GET_COMP(X, Y, Z, DIR)                          \
    hoodOld[FixedCoord<X, Y, Z>()].DIR()

#define SET_COMP(DIR)                           \
    hoodNew.DIR()

template<int DIM_X, int DIM_Y, int DIM_Z>
__global__ void benchmarkLBMSoA(int dimX, int dimY, int dimZ, double *gridOld, double *gridNew)
{
    int myX = blockIdx.x * blockDim.x + threadIdx.x + 2;
    int myY = blockIdx.y * blockDim.y + threadIdx.y + 2;
    int myZ = 2;

    int index = myZ * DIM_X * DIM_Y + myY * DIM_X + myX;
    int offset = DIM_X * DIM_Y;
    int end = DIM_X * DIM_Y * (dimZ - 2);

    LibFlatArray::soa_accessor_light<CellLBM, DIM_X, DIM_Y, DIM_Z, 0> hoodNew((char*)gridNew, index);
    LibFlatArray::soa_accessor_light<CellLBM, DIM_X, DIM_Y, DIM_Z, 0> hoodOldInternal((char*)gridOld, index);
    FixedNeighborhood<CellLBM, APITraits::SelectTopology<CellLBM>::Value, DIM_X, DIM_Y, DIM_Z, 0, LibFlatArray::soa_accessor_light> hoodOld(
        hoodOldInternal);

#pragma unroll 10
    for (; index < end; index += offset) {

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

        // density = rho;
        // velocityX = velX;
        // velocityY = velY;
        // velocityZ = velZ;

        const double dir_indep_trm = one_third*rho - 0.5*( velX*velX + velY*velY + velZ*velZ );

        SET_COMP(C)=omega_trm * GET_COMP(x,y,z,C) + omega_w0*( dir_indep_trm );

        SET_COMP(NW)=omega_trm * GET_COMP(x+1,y-1,z,NW) +
            omega_w2*( dir_indep_trm - ( velX-velY ) + 1.5*SQR( velX-velY ) );
        SET_COMP(SE)=omega_trm * GET_COMP(x-1,y+1,z,SE) +
            omega_w2*( dir_indep_trm + ( velX-velY ) + 1.5*SQR( velX-velY ) );
        SET_COMP(NE)=omega_trm * GET_COMP(x-1,y-1,z,NE) +
            omega_w2*( dir_indep_trm + ( velX+velY ) + 1.5*SQR( velX+velY ) );
        SET_COMP(SW)=omega_trm * GET_COMP(x+1,y+1,z,SW) +
            omega_w2*( dir_indep_trm - ( velX+velY ) + 1.5*SQR( velX+velY ) );

        SET_COMP(TW)=omega_trm * GET_COMP(x+1,y,z-1,TW) + omega_w2*( dir_indep_trm - ( velX-velZ ) + 1.5*SQR( velX-velZ ) );
        SET_COMP(BE)=omega_trm * GET_COMP(x-1,y,z+1,BE) + omega_w2*( dir_indep_trm + ( velX-velZ ) + 1.5*SQR( velX-velZ ) );
        SET_COMP(TE)=omega_trm * GET_COMP(x-1,y,z-1,TE) + omega_w2*( dir_indep_trm + ( velX+velZ ) + 1.5*SQR( velX+velZ ) );
        SET_COMP(BW)=omega_trm * GET_COMP(x+1,y,z+1,BW) + omega_w2*( dir_indep_trm - ( velX+velZ ) + 1.5*SQR( velX+velZ ) );

        SET_COMP(TS)=omega_trm * GET_COMP(x,y+1,z-1,TS) + omega_w2*( dir_indep_trm - ( velY-velZ ) + 1.5*SQR( velY-velZ ) );
        SET_COMP(BN)=omega_trm * GET_COMP(x,y-1,z+1,BN) + omega_w2*( dir_indep_trm + ( velY-velZ ) + 1.5*SQR( velY-velZ ) );
        SET_COMP(TN)=omega_trm * GET_COMP(x,y-1,z-1,TN) + omega_w2*( dir_indep_trm + ( velY+velZ ) + 1.5*SQR( velY+velZ ) );
        SET_COMP(BS)=omega_trm * GET_COMP(x,y+1,z+1,BS) + omega_w2*( dir_indep_trm - ( velY+velZ ) + 1.5*SQR( velY+velZ ) );

        SET_COMP(N)=omega_trm * GET_COMP(x,y-1,z,N) + omega_w1*( dir_indep_trm + velY + 1.5*SQR(velY));
        SET_COMP(S)=omega_trm * GET_COMP(x,y+1,z,S) + omega_w1*( dir_indep_trm - velY + 1.5*SQR(velY));
        SET_COMP(E)=omega_trm * GET_COMP(x-1,y,z,E) + omega_w1*( dir_indep_trm + velX + 1.5*SQR(velX));
        SET_COMP(W)=omega_trm * GET_COMP(x+1,y,z,W) + omega_w1*( dir_indep_trm - velX + 1.5*SQR(velX));
        SET_COMP(T)=omega_trm * GET_COMP(x,y,z-1,T) + omega_w1*( dir_indep_trm + velZ + 1.5*SQR(velZ));
        SET_COMP(B)=omega_trm * GET_COMP(x,y,z+1,B) + omega_w1*( dir_indep_trm - velZ + 1.5*SQR(velZ));
    }
}

#undef GET_COMP
#undef SET_COMP

template<int DIM_X, int DIM_Y, int DIM_Z>
class LBMSoA
{
public:
    static std::string family()
    {
        return "LBM";
    }

    static std::string species()
    {
        return "gold";
    }

    static void run(dim3 dimGrid, dim3 dimBlock, int dimX, int dimY, int dimZ, double *gridOld, double *gridNew)
    {
        benchmarkLBMSoA<DIM_X, DIM_Y, DIM_Z><<<dimGrid, dimBlock>>>(dimX, dimY, dimZ, gridOld, gridNew);
    }

    static int size()
    {
        return 20;
    }
};

template<int DIM_X, int DIM_Y, int DIM_Z>
class LBMClassic
{
public:
    static std::string family()
    {
        return "LBM";
    }

    static std::string species()
    {
        return "pepper";
    }

    static void run(dim3 dimGrid, dim3 dimBlock, int dimX, int dimY, int dimZ, double *gridOld, double *gridNew)
    {
        updateLBMClassic<DIM_X, DIM_Y, DIM_Z><<<dimGrid, dimBlock>>>(dimX, dimY, dimZ, gridOld, gridNew);
    }

    static int size()
    {
        return 20;
    }
};

template<int DIM_X, int DIM_Y, int DIM_Z>
class RTMSoA
{
public:
    static std::string family()
    {
        return "RTM";
    }

    static std::string species()
    {
        return "gold";
    }

    static void run(dim3 dimGrid, dim3 dimBlock, int dimX, int dimY, int dimZ, double *gridOld, double *gridNew)
    {
        updateRTMSoA<DIM_X, DIM_Y, DIM_Z><<<dimGrid, dimBlock>>>(dimX, dimY, dimZ, gridOld, gridNew);
    }

    static int size()
    {
        return 1;
    }
};

template<int DIM_X, int DIM_Y, int DIM_Z>
class RTMClassic
{
public:
    static std::string family()
    {
        return "RTM";
    }

    static std::string species()
    {
        return "pepper";
    }

    static void run(dim3 dimGrid, dim3 dimBlock, int dimX, int dimY, int dimZ, double *gridOld, double *gridNew)
    {
        updateRTMClassic<DIM_X, DIM_Y, DIM_Z><<<dimGrid, dimBlock>>>(dimX, dimY, dimZ, gridOld, gridNew);
    }

    static int size()
    {
        return 1;
    }
};

template<template<int A, int B, int C> class KERNEL, int DIM_X, int DIM_Y, int DIM_Z>
double benchmarkCUDA(int dimX, int dimY, int dimZ, int repeats)
{
    std::size_t size = DIM_X * DIM_Y * (DIM_Z + 4) * KERNEL<0, 0, 0>::size();
    std::size_t bytesize = size * sizeof(double);

    std::vector<double> grid(size, 4711);

    double *devGridOld;
    double *devGridNew;
    cudaMalloc(&devGridOld, bytesize);
    cudaMalloc(&devGridNew, bytesize);
    CUDAUtil::checkForError();

    cudaMemcpy(devGridOld, &grid[0], bytesize, cudaMemcpyHostToDevice);
    cudaMemcpy(devGridNew, &grid[0], bytesize, cudaMemcpyHostToDevice);

    int blockWidth = 1;
    for (; blockWidth <= dimX; blockWidth *= 2) {
    }
    blockWidth /= 2;
    blockWidth = std::min(256, blockWidth);

    dim3 dimBlock(blockWidth, 2, 1);
    dim3 dimGrid(dimX / dimBlock.x, dimY / dimBlock.y, 1);
    cudaDeviceSynchronize();

    double seconds = 0;
    {
        ScopedTimer t(&seconds);

        for (int t = 0; t < repeats; ++t) {
            KERNEL<DIM_X, DIM_Y, DIM_Z>::run(dimGrid, dimBlock, dimX, dimY, dimZ, devGridOld, devGridNew);
            std::swap(devGridOld, devGridNew);
        }
        cudaDeviceSynchronize();
    }

    CUDAUtil::checkForError();

    cudaMemcpy(&grid[0], devGridNew, bytesize, cudaMemcpyDeviceToHost);
    cudaFree(devGridOld);
    cudaFree(devGridNew);

    double updates = 1.0 * dimGrid.x * dimBlock.x * dimGrid.y * dimBlock.y * dimZ * repeats;
    double glups = 1e-9 * updates / seconds;
    return glups;
}

template<template<int A, int B, int C> class KERNEL>
class BenchmarkCUDA : public GPUBenchmark
{
public:
    std::string family()
    {
        return KERNEL<0, 0, 0>::family();
    }

    std::string species()
    {
        return KERNEL<0, 0, 0>::species();
    }

    std::string unit()
    {
        return "GLUPS";
    }

    double performance2(const Coord<3>& dim)
    {
#define CASE(DIM, ADD)                                                  \
        if (max(dim) <= DIM) {                                          \
            return benchmarkCUDA<KERNEL, DIM + ADD, DIM, DIM>(          \
                dim.x(), dim.y(), dim.z(), 20);                         \
        }

        CASE(32,  0);
        CASE(64,  0);
        CASE(96,  0);
        CASE(128, 0);
        CASE(160, 0);
        CASE(192, 0);
        CASE(256, 0);
        CASE(288, 0);
        CASE(320, 0);
        CASE(352, 0);
        CASE(384, 0);
        CASE(416, 0);
        CASE(448, 0);
        CASE(480, 0);
        CASE(512, 0);
        CASE(544, 0);

#undef CASE

        throw std::range_error("dim too large");
    }

    int max(const Coord<3>& coord) const
    {
        return (std::max)(coord.x(), (std::max)(coord.y(), coord.z()));
    }
};

void cudaTests(std::string revision, bool quick, int cudaDevice)
{
    cudaSetDevice(cudaDevice);
    LibFlatArray::evaluate eval(revision);

    for (int d = 32; d <= 544; d += 4) {
        eval(BenchmarkCUDA<RTMClassic>(), toVector(Coord<3>::diagonal(d)));
    }
    for (int d = 32; d <= 544; d += 4) {
        eval(BenchmarkCUDA<RTMSoA>(),     toVector(Coord<3>::diagonal(d)));
    }
    for (int d = 32; d <= 160; d += 4) {
        Coord<3> dim(d, d, 256 + 32 - 4);
        eval(BenchmarkCUDA<LBMClassic>(), toVector(Coord<3>::diagonal(d)));
    }
    for (int d = 32; d <= 160; d += 4) {
        Coord<3> dim(d, d, 256 + 32 - 4);
        eval(BenchmarkCUDA<LBMSoA>(),     toVector(Coord<3>::diagonal(d)));
    }
}
