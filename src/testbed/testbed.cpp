#include <cmath>
#include <typeinfo> 
// #include <CL/cl.h>
#include <iostream>
#include <emmintrin.h>
// #include <pmmintrin.h>
#include <sys/time.h>
#include <vector>
#include <libgeodecomp/misc/grid.h>

using namespace LibGeoDecomp;

class Scalar2D
{
public:
    static int coefficients()
    {
        return 0;
    }

    inline void step(double *src, double *dst, int offset, int startX, int endX)
    {
        for (int x = startX; x < endX; ++x) {
            dst[x] = (src[x - offset] + src[x - 1] + src[x] + src[x + 1] + src[x + offset]) * 0.2;
        }
    }

    int flops()
    {
        return 5;
    }
};

class VectorizedSSEMelbourneShuffle2D
{
public:
    static int coefficients()
    {
        return 0;
    }

    inline void step(double *src, double *dst, int offset, int startX, int endX)
    {
        int x = startX;
        Scalar2D scalarUpdater;

        if ((x & 1) == 1) {
            scalarUpdater.step(src, dst, offset, x, x + 1);
            x += 1;
        }

        __m128d oneFifth = _mm_set_pd(1.0/3.0, 1.0/3.0);
        __m128d buff0 = _mm_loadu_pd(src + x - 1);
        __m128d same0 = _mm_load_pd(src + x + 0);

        int paddedEndX = endX - 7;
        for (; x < paddedEndX; x += 8) {
            // load center row
            __m128d same1 = _mm_load_pd(src + x + 2);
            __m128d same2 = _mm_load_pd(src + x + 4);
            __m128d same3 = _mm_load_pd(src + x + 6);
            __m128d same4 = _mm_load_pd(src + x + 8);
            
            // shuffle values obtain left/right neighbors
            __m128d buff1 = _mm_shuffle_pd(same0, same1, (1 << 0) | (0 << 2));
            __m128d buff2 = _mm_shuffle_pd(same1, same2, (1 << 0) | (0 << 2));
            __m128d buff3 = _mm_shuffle_pd(same2, same3, (1 << 0) | (0 << 2));
            __m128d buff4 = _mm_shuffle_pd(same3, same4, (1 << 0) | (0 << 2));

            // load top row
            __m128d temp0 = _mm_load_pd(src - offset + x + 0);
            __m128d temp1 = _mm_load_pd(src - offset + x + 2);
            __m128d temp2 = _mm_load_pd(src - offset + x + 4);
            __m128d temp3 = _mm_load_pd(src - offset + x + 6);

            // add center row with left...
            same0 = _mm_add_pd(same0, buff0);
            same1 = _mm_add_pd(same1, buff1);
            same2 = _mm_add_pd(same2, buff2);
            same3 = _mm_add_pd(same3, buff3);

            // ...and right neighbors
            same0 = _mm_add_pd(same0, buff1);
            same1 = _mm_add_pd(same1, buff2);
            same2 = _mm_add_pd(same2, buff3);
            same3 = _mm_add_pd(same3, buff4);
    
            // load bottom row
            buff0 = _mm_load_pd(src + offset + x + 0);
            buff1 = _mm_load_pd(src + offset + x + 2);
            buff2 = _mm_load_pd(src + offset + x + 4);
            buff3 = _mm_load_pd(src + offset + x + 6);
        
            // add top row
            same0 = _mm_add_pd(same0, temp0);
            same1 = _mm_add_pd(same1, temp1);
            same2 = _mm_add_pd(same2, temp2);
            same3 = _mm_add_pd(same3, temp3);

            // add bottom row
            same0 = _mm_add_pd(same0, buff0);
            same1 = _mm_add_pd(same1, buff1);
            same2 = _mm_add_pd(same2, buff2);
            same3 = _mm_add_pd(same3, buff3);

            // scale down...
            same0 = _mm_mul_pd(same0, oneFifth);
            same1 = _mm_mul_pd(same1, oneFifth);
            same2 = _mm_mul_pd(same2, oneFifth);
            same3 = _mm_mul_pd(same3, oneFifth);

            // ...and store
            _mm_store_pd(dst + 0, same0);
            _mm_store_pd(dst + 2, same1);
            _mm_store_pd(dst + 4, same2);
            _mm_store_pd(dst + 6, same3);

            same0 = same4;
            buff0 = buff4;
        }

        scalarUpdater.step(src, dst, offset, x, endX);
    }

    int flops()
    {
        return 5;
    }
};

template<int DIM_X=0, int DIM_Y=0, int DIM_Z=0>
class FixedCoord
{
public:
};

template<int DIM_X=0, int DIM_Y=0, int DIM_Z=0>
class Neighborhood
{
public:
    Neighborhood(double *_coefficients, double *_source, const int& _index) :
        coefficients(_coefficients),
        source(_source),
        index(_index)
    {}

    Neighborhood
    operator[](Coord<3> coord) const
    {
        return Neighborhood<DIM_X, DIM_Y, DIM_Z>(
            coefficients, 
            source,
            coord.x() +
            coord.y() * DIM_X + 
            coord.z() * DIM_X * DIM_Y);
    }

    const double& src(const int& x = 0) const
    {
        return source[index + x];
    }

    template<int C> 
    const double& coeff(const FixedCoord<C, 0, 0> /*unused*/, const int& x) const
    { 
        return coefficients[C * DIM_X * DIM_Y * DIM_Z + index + x];
    }

private:
    double *coefficients;
    double *source;
    int index;
};

template<int DIM_X, int DIM_Y, int DIM_Z, int INDEX=0>
class FixedNeighborhood
{
public:
    FixedNeighborhood(double *_coefficients, double *_source) :
        coefficients(_coefficients),
        source(_source)
    {}

    // template<int X, int Y, int Z, int C> 
    // const double& coeff(const int& x) const
    // {
    //     return coefficients[C * DIM_X * DIM_Y * DIM_Z + Z * DIM_X * DIM_Y + Y * DIM_X + X + x];
    // }

    // template<int X, int Y, int Z> 
    // const double& src(const int& x) const
    // {
    //     return source[Z * DIM_X * DIM_Y + Y * DIM_X + X + x];
    // }

    template<int X, int Y, int Z>
    FixedNeighborhood<DIM_X, DIM_Y, DIM_Z, X + Y * DIM_X + Z * DIM_X * DIM_Y>
    operator[](FixedCoord<X, Y, Z> /*unused*/) const
    {
        return FixedNeighborhood<DIM_X, DIM_Y, DIM_Z, X + Y * DIM_X + Z * DIM_X * DIM_Y>(coefficients, source);
    }

    const double& src(const int& x) const
    {
        return source[INDEX + x];
    }

    template<int C> 
    const double& coeff(const FixedCoord<C, 0, 0> /*unused*/, const int& x) const
    { 
        return coefficients[C * DIM_X * DIM_Y * DIM_Z + INDEX + x];
    }

private:
    double *coefficients;
    double *source;
};

template<typename UPDATER, int DIM>
class Benchmark
{
public:
    typedef Grid<double, typename Topologies::Cube<DIM>::Topology> GridType;


    void run(Coord<DIM> dim, int repeats)
    {
        Coord<DIM> coeffDim = dim;
        coeffDim.c[DIM - 1] *= UPDATER::coefficients();
        GridType coeff(coeffDim, 0.1);
        GridType a(dim, 1.0);
        GridType b(dim, 1.0);

        GridType *oldGrid = &a;
        GridType *newGrid = &b;

        UPDATER updater;

        long long tStart = getUTtime();
        std::vector<double*> coefficients(UPDATER::coefficients());

        for (int t = 0; t < repeats; ++t) {
            for (int z = 1; z < dim.c[2] - 1; ++z) {
                for (int y = 1; y < dim.c[1] - 1; ++y) {
                    Coord<DIM> c(0, y, z);
                    Coord<DIM> coeffCoord = c;

                    // Defiant
                    updater.step(
                        FixedNeighborhood<4, 4, MY_SIZE>(&coeff.at(coeffCoord), &oldGrid->at(c)),
                        &newGrid->at(c), 
                        1, 
                        dim.c[0] - 1);

                    // updater.step(
                    //     Neighborhood<4, 4, MY_SIZE>(&coeff.at(coeffCoord), &oldGrid->at(c)),
                    //     &newGrid->at(c), 
                    //     dim.c[0], 
                    //     dim.c[0] * dim.c[1], 
                    //     1, 
                    //     dim.c[0] - 1);
                    
                    // for (int i = 0; i < UPDATER::coefficients(); ++i) {
                    //     coefficients[i] = &coeff.at(coeffCoord);
                    //     coeffCoord.c[DIM - 1] += dim.c[DIM - 1];
                    // }
                    // updater.step(
                    //     &coefficients[0], 
                    //     &oldGrid->at(c), 
                    //     &newGrid->at(c), 
                    //     dim.c[0], 
                    //     dim.c[0] * dim.c[1], 
                    //     1, 
                    //     dim.c[0] - 1);

                    // double *source[9] = {
                    //     &oldGrid->at(Coord<DIM>(0, y - 1, z + 1)),
                    //     &oldGrid->at(Coord<DIM>(0, y - 1, z    )),
                    //     &oldGrid->at(Coord<DIM>(0, y - 1, z - 1)),
                    //     &oldGrid->at(Coord<DIM>(0, y,     z + 1)),
                    //     &oldGrid->at(Coord<DIM>(0, y,     z    )),
                    //     &oldGrid->at(Coord<DIM>(0, y,     z - 1)),
                    //     &oldGrid->at(Coord<DIM>(0, y + 1, z + 1)),
                    //     &oldGrid->at(Coord<DIM>(0, y + 1, z    )),
                    //     &oldGrid->at(Coord<DIM>(0, y + 1, z - 1))
                    // };
                    // updater.update(source, &newGrid->at(c), 1, dim.c[0] - 1);
                }
            }
            std::swap(newGrid, oldGrid);
        }

        long long tEnd = getUTtime();
        evaluate(dim, repeats, tEnd - tStart);
    }

    void exercise() 
    {
        std::cout << "# " << typeid(UPDATER).name() << "\n";
        int lastDim = 0;
        for (int i = 4; i <= 4096; i *= 2) {
            int intermediateSteps = 8;
            for (int j = 0; j < intermediateSteps; ++j) {
                int d = i * std::pow(2, j * (1.0 / intermediateSteps));
                if (d % 2) {
                    d += 1;
                }

                if (d > lastDim) {
                    lastDim = d;
                    Coord<DIM> dim;
                    dim.c[0] = d;
                    dim.c[1] = 4;
                    dim.c[2] = 4;
                    // for (int i = 0; i < DIM; ++i)
                    //     dim.c[i] = d;
                    int repeats = std::max(1, 10000000 / dim.prod());
                    run(dim, repeats);
                }
            }
        }
        std::cout << "\n";
    }

private:
    long long getUTtime()
    {
        timeval t;
        gettimeofday(&t, 0);
        return (long long)t.tv_sec * 1000000 + t.tv_usec;
    }

    void evaluate(Coord<DIM> dim, int repeats, long long uTime)
    {
        double seconds = 1.0 * uTime / 1000 / 1000;
        Coord<DIM> inner = dim;
        for (int i = 0; i < DIM; ++i)
            inner.c[i] -= 2;
        double gflops = 1.0 * UPDATER().flops() * inner.prod() * 
            repeats / 1000 / 1000 / 1000 / seconds;
        std::cout << dim.x() << " " << gflops << "\n";
    }


};

#define TN 0
#define T  1
#define TS 2
#define N  3
#define C  4
#define S  5
#define BN 6
#define B  7
#define BS 8

class Jacobi3D
{
public:
    static int coefficients()
    {
        return 1;
    }

    inline void update(double **src, double *dst, int startX, int endX)
    {
       int x = startX;

       if ((x & 1) == 1) {
           updateScalar(src, dst, x, x + 1);
           x += 1;
       }

       __m128d oneSeventh = _mm_set_pd(1.0/7.0, 1.0/7.0);
       __m128d buff0 = _mm_loadu_pd(src[C] + x - 1);
       __m128d same0 = _mm_load_pd(src[C] + x + 0);

       int paddedEndX = endX - 7;
       for (; x < paddedEndX; x += 8) {
           // load center row
           __m128d same1 = _mm_load_pd(src[0] + x + 2);
           __m128d same2 = _mm_load_pd(src[0] + x + 4);
           __m128d same3 = _mm_load_pd(src[0] + x + 6);
           __m128d same4 = _mm_load_pd(src[0] + x + 8);
            
           // shuffle values obtain left/right neighbors
           __m128d buff1 = _mm_shuffle_pd(same0, same1, (1 << 0) | (0 << 2));
           __m128d buff2 = _mm_shuffle_pd(same1, same2, (1 << 0) | (0 << 2));
           __m128d buff3 = _mm_shuffle_pd(same2, same3, (1 << 0) | (0 << 2));
           __m128d buff4 = _mm_shuffle_pd(same3, same4, (1 << 0) | (0 << 2));
   
           // load top row
           __m128d temp0 = _mm_load_pd(src[T] + x + 0);
           __m128d temp1 = _mm_load_pd(src[T] + x + 2);
           __m128d temp2 = _mm_load_pd(src[T] + x + 4);
           __m128d temp3 = _mm_load_pd(src[T] + x + 6);

           // add center row with left...
           same0 = _mm_add_pd(same0, buff0);
           same1 = _mm_add_pd(same1, buff1);
           same2 = _mm_add_pd(same2, buff2);
           same3 = _mm_add_pd(same3, buff3);

           // ...and right neighbors
           same0 = _mm_add_pd(same0, buff1);
           same1 = _mm_add_pd(same1, buff2);
           same2 = _mm_add_pd(same2, buff3);
           same3 = _mm_add_pd(same3, buff4);
    
           // load bottom row
           buff0 = _mm_load_pd(src[B] + x + 0);
           buff1 = _mm_load_pd(src[B] + x + 2);
           buff2 = _mm_load_pd(src[B] + x + 4);
           buff3 = _mm_load_pd(src[B] + x + 6);
        
           // add top row
           same0 = _mm_add_pd(same0, temp0);
           same1 = _mm_add_pd(same1, temp1);
           same2 = _mm_add_pd(same2, temp2);
           same3 = _mm_add_pd(same3, temp3);

           // load north row
           temp0 = _mm_load_pd(src[N] + x + 0);
           temp1 = _mm_load_pd(src[N] + x + 2);
           temp2 = _mm_load_pd(src[N] + x + 4);
           temp3 = _mm_load_pd(src[N] + x + 6);

           // add bottom row
           same0 = _mm_add_pd(same0, buff0);
           same1 = _mm_add_pd(same1, buff1);
           same2 = _mm_add_pd(same2, buff2);
           same3 = _mm_add_pd(same3, buff3);
            
           // load south row
           buff0 = _mm_load_pd(src[S] + x + 0);
           buff1 = _mm_load_pd(src[S] + x + 2);
           buff2 = _mm_load_pd(src[S] + x + 4);
           buff3 = _mm_load_pd(src[S] + x + 6);

           // add north row
           same0 = _mm_add_pd(same0, temp0);
           same1 = _mm_add_pd(same1, temp1);
           same2 = _mm_add_pd(same2, temp2);
           same3 = _mm_add_pd(same3, temp3);

           // add south row
           same0 = _mm_add_pd(same0, buff0);
           same1 = _mm_add_pd(same1, buff1);
           same2 = _mm_add_pd(same2, buff2);
           same3 = _mm_add_pd(same3, buff3);

           // scale down...
           same0 = _mm_mul_pd(same0, oneSeventh);
           same1 = _mm_mul_pd(same1, oneSeventh);
           same2 = _mm_mul_pd(same2, oneSeventh);
           same3 = _mm_mul_pd(same3, oneSeventh);

           // ...and store
           _mm_store_pd(dst + x + 0, same0);
           _mm_store_pd(dst + x + 2, same1);
           _mm_store_pd(dst + x + 4, same2);
           _mm_store_pd(dst + x + 6, same3);

           same0 = same4;
           buff0 = buff4;
       }

       updateScalar(src, dst, x, endX);
    }

    inline void updateScalar(double *src[9], double *dst, int startX, int endX)
    {
        for (int x = startX; x < endX; ++x) {
            dst[x] = 
                (src[S][x] +
                 src[T][x] +
                 src[C][x - 1] +
                 src[C][x] +
                 src[C][x + 1] +
                 src[B][x] +
                 src[N][x]) * (1.0 / 7.0);
        }
    }

    int flops()
    {
        return 8;
    }
};


class Scalar3D
{
public:
    static int coefficients()
    {
        return 7;
    }

    inline void step(double *coeff[7], double *src, double *dst, int offsetY, int offsetZ, int startX, int endX)
    {
        for (int x = startX; x < endX; ++x) {
            dst[x] = 
                coeff[0][x] * src[x - offsetZ] +
                coeff[1][x] * src[x - offsetY] +
                coeff[2][x] * src[x - 1] +
                coeff[3][x] * src[x] +
                coeff[4][x] * src[x + 1] +
                coeff[5][x] * src[x + offsetY] +
                coeff[6][x] * src[x + offsetZ];
        }
    }

    int flops()
    {
        return 13;
    }
};

class Vectorized3D
{
public:
    static int coefficients()
    {
        return 7;
    }

    inline void step(double **coeff, double *src, double *dst, int offsetY, int offsetZ, int startX, int endX)
    {
        int x = startX;
        Scalar3D scalarUpdater;

        if ((x & 1) == 1) {
            scalarUpdater.step(coeff, src, dst, offsetY, offsetZ, x, x + 1);
            x += 1;
        }

        __m128d same0 = _mm_load_pd(src + x + 0);
        __m128d neig0 = _mm_loadu_pd(src + x + 1);
        
        int paddedEndX = endX - 7;
        for (; x < paddedEndX; x += 8) {
            __m128d same1 = _mm_load_pd(src + x + 2);
            __m128d same2 = _mm_load_pd(src + x + 4);
            __m128d same3 = _mm_load_pd(src + x + 6);
            __m128d same4 = _mm_load_pd(src + x + 8);

            __m128d neig1 = _mm_shuffle_pd(same0, same1, (1 << 0) | (0 << 2));
            __m128d neig2 = _mm_shuffle_pd(same1, same2, (1 << 0) | (0 << 2));
            __m128d neig3 = _mm_shuffle_pd(same2, same3, (1 << 0) | (0 << 2));
            __m128d neig4 = _mm_shuffle_pd(same3, same4, (1 << 0) | (0 << 2));

            same0 = _mm_mul_pd(same0, _mm_load_pd(&coeff[3][x + 0]));
            same1 = _mm_mul_pd(same1, _mm_load_pd(&coeff[3][x + 2]));
            same2 = _mm_mul_pd(same2, _mm_load_pd(&coeff[3][x + 4]));
            same3 = _mm_mul_pd(same3, _mm_load_pd(&coeff[3][x + 6]));

            __m128d temp1 = _mm_mul_pd(neig0, _mm_load_pd(&coeff[2][x + 0]));
            __m128d temp2 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[2][x + 2]));
            __m128d temp3 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[2][x + 4]));
            __m128d temp4 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[2][x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            temp1 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[4][x + 0]));
            temp2 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[4][x + 2]));
            temp3 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[4][x + 4]));
            temp4 = _mm_mul_pd(neig4, _mm_load_pd(&coeff[4][x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            neig0 = _mm_load_pd(src + x - offsetZ + 0);
            neig1 = _mm_load_pd(src + x - offsetZ + 2);
            neig2 = _mm_load_pd(src + x - offsetZ + 4);
            neig3 = _mm_load_pd(src + x - offsetZ + 6);

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&coeff[0][x + 0]));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[0][x + 2]));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[0][x + 4]));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[0][x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            neig0 = _mm_load_pd(src + x - offsetY + 0);
            neig1 = _mm_load_pd(src + x - offsetY + 2);
            neig2 = _mm_load_pd(src + x - offsetY + 4);
            neig3 = _mm_load_pd(src + x - offsetY + 6);

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&coeff[1][x + 0]));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[1][x + 2]));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[1][x + 4]));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[1][x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            neig0 = _mm_load_pd(src + x + offsetY + 0);
            neig1 = _mm_load_pd(src + x + offsetY + 2);
            neig2 = _mm_load_pd(src + x + offsetY + 4);
            neig3 = _mm_load_pd(src + x + offsetY + 6);

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&coeff[5][x + 0]));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[5][x + 2]));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[5][x + 4]));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[5][x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            neig0 = _mm_load_pd(src + x + offsetZ + 0);
            neig1 = _mm_load_pd(src + x + offsetZ + 2);
            neig2 = _mm_load_pd(src + x + offsetZ + 4);
            neig3 = _mm_load_pd(src + x + offsetZ + 6);

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&coeff[6][x + 0]));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[6][x + 2]));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[6][x + 4]));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[6][x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            _mm_store_pd(dst + 0, same0);
            _mm_store_pd(dst + 2, same1);
            _mm_store_pd(dst + 4, same2);
            _mm_store_pd(dst + 6, same3);

            same0 = same4;
            neig0 = neig4;

            // dst[x] = 
            //     coeff[0][x] * src[x - offsetZ] +
            //     coeff[1][x] * src[x - offsetY] +
            //     coeff[2][x] * src[x - 1] +
            //     coeff[3][x] * src[x] +
            //     coeff[4][x] * src[x + 1] +
            //     coeff[5][x] * src[x + offsetY] +
            //     coeff[6][x] * src[x + offsetZ];
        }

        scalarUpdater.step(coeff, src, dst, offsetY, offsetZ, x, endX);
    }

    int flops()
    {
        return 13;
    }
};

class ExtendedScalar3D
{
public:
    static int coefficients()
    {
        return 13;
    }

    inline void step(double *coeff[13], double *src, double *dst, int offsetY, int offsetZ, int startX, int endX)
    {
        for (int x = startX; x < endX; ++x) {
            dst[x] = 
                coeff[ 0][x] * src[x - offsetZ - offsetY] +
                coeff[ 1][x] * src[x - offsetZ] +
                coeff[ 2][x] * src[x - offsetZ + offsetY] +
                coeff[ 3][x] * src[x - offsetY] +
                coeff[ 4][x] * src[x - 1] +
                coeff[ 5][x] * src[x] +
                coeff[ 6][x] * src[x + 1] +
                coeff[ 7][x] * src[x + offsetY] +
                coeff[ 8][x] * src[x + offsetZ - offsetY] +
                coeff[ 9][x] * src[x + offsetZ] +
                coeff[10][x] * src[x + offsetZ + offsetY] +
                coeff[11][x - offsetZ - offsetY] +
                coeff[11][x - offsetZ] +
                coeff[11][x - offsetZ + offsetY] +
                coeff[11][x - offsetY] +
                coeff[11][x] +
                coeff[11][x + offsetY] +
                coeff[11][x + offsetZ - offsetY] +
                coeff[11][x + offsetZ] +
                coeff[11][x + offsetZ + offsetY] +
                coeff[12][x - offsetZ - offsetY] +
                coeff[12][x - offsetZ] +
                coeff[12][x - offsetZ + offsetY] +
                coeff[12][x - offsetY] +
                coeff[12][x] +
                coeff[12][x + offsetY] +
                coeff[12][x + offsetZ - offsetY] +
                coeff[12][x + offsetZ] +
                coeff[12][x + offsetZ + offsetY];
        }
    }

    int flops()
    {
        return 40;
    }
};

class ExtendedVectorized3D
{
public:
    static int coefficients()
    {
        return 13;
    }

    inline void step(double *coeff[13], double *src, double *dst, int offsetY, int offsetZ, int startX, int endX)
    {
        int x = startX;
        ExtendedScalar3D scalarUpdater;

        if ((x & 1) == 1) {
            scalarUpdater.step(coeff, src, dst, offsetY, offsetZ, x, x + 1);
            x += 1;
        }

        __m128d same0 = _mm_load_pd(src + x + 0);
        __m128d neig0 = _mm_loadu_pd(src + x + 1);
        
        int paddedEndX = endX - 7;
        for (; x < paddedEndX; x += 8) {
            __m128d same1 = _mm_load_pd(src + x + 2);
            __m128d same2 = _mm_load_pd(src + x + 4);
            __m128d same3 = _mm_load_pd(src + x + 6);
            __m128d same4 = _mm_load_pd(src + x + 8);

            __m128d neig1 = _mm_shuffle_pd(same0, same1, (1 << 0) | (0 << 2));
            __m128d neig2 = _mm_shuffle_pd(same1, same2, (1 << 0) | (0 << 2));
            __m128d neig3 = _mm_shuffle_pd(same2, same3, (1 << 0) | (0 << 2));
            __m128d neig4 = _mm_shuffle_pd(same3, same4, (1 << 0) | (0 << 2));

            same0 = _mm_mul_pd(same0, _mm_load_pd(&coeff[3][x + 0]));
            same1 = _mm_mul_pd(same1, _mm_load_pd(&coeff[3][x + 2]));
            same2 = _mm_mul_pd(same2, _mm_load_pd(&coeff[3][x + 4]));
            same3 = _mm_mul_pd(same3, _mm_load_pd(&coeff[3][x + 6]));

            __m128d temp1 = _mm_mul_pd(neig0, _mm_load_pd(&coeff[2][x + 0]));
            __m128d temp2 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[2][x + 2]));
            __m128d temp3 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[2][x + 4]));
            __m128d temp4 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[2][x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            temp1 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[4][x + 0]));
            temp2 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[4][x + 2]));
            temp3 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[4][x + 4]));
            temp4 = _mm_mul_pd(neig4, _mm_load_pd(&coeff[4][x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            neig0 = _mm_load_pd(src + x - offsetZ + 0);
            neig1 = _mm_load_pd(src + x - offsetZ + 2);
            neig2 = _mm_load_pd(src + x - offsetZ + 4);
            neig3 = _mm_load_pd(src + x - offsetZ + 6);

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&coeff[0][x + 0]));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[0][x + 2]));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[0][x + 4]));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[0][x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            neig0 = _mm_load_pd(src + x - offsetY + 0);
            neig1 = _mm_load_pd(src + x - offsetY + 2);
            neig2 = _mm_load_pd(src + x - offsetY + 4);
            neig3 = _mm_load_pd(src + x - offsetY + 6);

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&coeff[1][x + 0]));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[1][x + 2]));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[1][x + 4]));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[1][x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            neig0 = _mm_load_pd(src + x + offsetY + 0);
            neig1 = _mm_load_pd(src + x + offsetY + 2);
            neig2 = _mm_load_pd(src + x + offsetY + 4);
            neig3 = _mm_load_pd(src + x + offsetY + 6);

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&coeff[5][x + 0]));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[5][x + 2]));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[5][x + 4]));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[5][x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(src + x + offsetZ + 0);
            neig1 = _mm_load_pd(src + x + offsetZ + 2);
            neig2 = _mm_load_pd(src + x + offsetZ + 4);
            neig3 = _mm_load_pd(src + x + offsetZ + 6);

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&coeff[6][x + 0]));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[6][x + 2]));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[6][x + 4]));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[6][x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(src + x - offsetZ - offsetY + 0);
            neig1 = _mm_load_pd(src + x - offsetZ - offsetY + 2);
            neig2 = _mm_load_pd(src + x - offsetZ - offsetY + 4);
            neig3 = _mm_load_pd(src + x - offsetZ - offsetY + 6);

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&coeff[7][x + 0]));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[7][x + 2]));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[7][x + 4]));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[7][x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(src + x - offsetZ + offsetY + 0);
            neig1 = _mm_load_pd(src + x - offsetZ + offsetY + 2);
            neig2 = _mm_load_pd(src + x - offsetZ + offsetY + 4);
            neig3 = _mm_load_pd(src + x - offsetZ + offsetY + 6);

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&coeff[8][x + 0]));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[8][x + 2]));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[8][x + 4]));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[8][x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(src + x + offsetZ - offsetY + 0);
            neig1 = _mm_load_pd(src + x + offsetZ - offsetY + 2);
            neig2 = _mm_load_pd(src + x + offsetZ - offsetY + 4);
            neig3 = _mm_load_pd(src + x + offsetZ - offsetY + 6);

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&coeff[9][x + 0]));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[9][x + 2]));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[9][x + 4]));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[9][x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(src + x + offsetZ + offsetY + 0);
            neig1 = _mm_load_pd(src + x + offsetZ + offsetY + 2);
            neig2 = _mm_load_pd(src + x + offsetZ + offsetY + 4);
            neig3 = _mm_load_pd(src + x + offsetZ + offsetY + 6);

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&coeff[10][x + 0]));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[10][x + 2]));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[10][x + 4]));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[10][x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[11] + x - offsetZ - offsetY + 0);
            neig1 = _mm_load_pd(coeff[11] + x - offsetZ - offsetY + 2);
            neig2 = _mm_load_pd(coeff[11] + x - offsetZ - offsetY + 4);
            neig3 = _mm_load_pd(coeff[11] + x - offsetZ - offsetY + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[11] + x - offsetZ + 0);
            neig1 = _mm_load_pd(coeff[11] + x - offsetZ + 2);
            neig2 = _mm_load_pd(coeff[11] + x - offsetZ + 4);
            neig3 = _mm_load_pd(coeff[11] + x - offsetZ + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[11] + x - offsetZ + offsetY + 0);
            neig1 = _mm_load_pd(coeff[11] + x - offsetZ + offsetY + 2);
            neig2 = _mm_load_pd(coeff[11] + x - offsetZ + offsetY + 4);
            neig3 = _mm_load_pd(coeff[11] + x - offsetZ + offsetY + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[11] + x - offsetY + 0);
            neig1 = _mm_load_pd(coeff[11] + x - offsetY + 2);
            neig2 = _mm_load_pd(coeff[11] + x - offsetY + 4);
            neig3 = _mm_load_pd(coeff[11] + x - offsetY + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[11] + offsetY + x + 0);
            neig1 = _mm_load_pd(coeff[11] + offsetY + x + 2);
            neig2 = _mm_load_pd(coeff[11] + offsetY + x + 4);
            neig3 = _mm_load_pd(coeff[11] + offsetY + x + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[11] + x + 0);
            neig1 = _mm_load_pd(coeff[11] + x + 2);
            neig2 = _mm_load_pd(coeff[11] + x + 4);
            neig3 = _mm_load_pd(coeff[11] + x + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[11] + x + offsetZ - offsetY + 0);
            neig1 = _mm_load_pd(coeff[11] + x + offsetZ - offsetY + 2);
            neig2 = _mm_load_pd(coeff[11] + x + offsetZ - offsetY + 4);
            neig3 = _mm_load_pd(coeff[11] + x + offsetZ - offsetY + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[11] + x + offsetZ + 0);
            neig1 = _mm_load_pd(coeff[11] + x + offsetZ + 2);
            neig2 = _mm_load_pd(coeff[11] + x + offsetZ + 4);
            neig3 = _mm_load_pd(coeff[11] + x + offsetZ + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[11] + x + offsetZ - offsetY + 0);
            neig1 = _mm_load_pd(coeff[11] + x + offsetZ - offsetY + 2);
            neig2 = _mm_load_pd(coeff[11] + x + offsetZ - offsetY + 4);
            neig3 = _mm_load_pd(coeff[11] + x + offsetZ - offsetY + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[11] + x + offsetZ + offsetY + 0);
            neig1 = _mm_load_pd(coeff[11] + x + offsetZ + offsetY + 2);
            neig2 = _mm_load_pd(coeff[11] + x + offsetZ + offsetY + 4);
            neig3 = _mm_load_pd(coeff[11] + x + offsetZ + offsetY + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[12] + x - offsetZ - offsetY + 0);
            neig1 = _mm_load_pd(coeff[12] + x - offsetZ - offsetY + 2);
            neig2 = _mm_load_pd(coeff[12] + x - offsetZ - offsetY + 4);
            neig3 = _mm_load_pd(coeff[12] + x - offsetZ - offsetY + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[12] + x - offsetZ + 0);
            neig1 = _mm_load_pd(coeff[12] + x - offsetZ + 2);
            neig2 = _mm_load_pd(coeff[12] + x - offsetZ + 4);
            neig3 = _mm_load_pd(coeff[12] + x - offsetZ + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[12] + x - offsetZ + offsetY + 0);
            neig1 = _mm_load_pd(coeff[12] + x - offsetZ + offsetY + 2);
            neig2 = _mm_load_pd(coeff[12] + x - offsetZ + offsetY + 4);
            neig3 = _mm_load_pd(coeff[12] + x - offsetZ + offsetY + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[12] + x - offsetY + 0);
            neig1 = _mm_load_pd(coeff[12] + x - offsetY + 2);
            neig2 = _mm_load_pd(coeff[12] + x - offsetY + 4);
            neig3 = _mm_load_pd(coeff[12] + x - offsetY + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[12] + offsetY + x + 0);
            neig1 = _mm_load_pd(coeff[12] + offsetY + x + 2);
            neig2 = _mm_load_pd(coeff[12] + offsetY + x + 4);
            neig3 = _mm_load_pd(coeff[12] + offsetY + x + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[12] + x + 0);
            neig1 = _mm_load_pd(coeff[12] + x + 2);
            neig2 = _mm_load_pd(coeff[12] + x + 4);
            neig3 = _mm_load_pd(coeff[12] + x + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[12] + x + offsetZ - offsetY + 0);
            neig1 = _mm_load_pd(coeff[12] + x + offsetZ - offsetY + 2);
            neig2 = _mm_load_pd(coeff[12] + x + offsetZ - offsetY + 4);
            neig3 = _mm_load_pd(coeff[12] + x + offsetZ - offsetY + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[12] + x + offsetZ + 0);
            neig1 = _mm_load_pd(coeff[12] + x + offsetZ + 2);
            neig2 = _mm_load_pd(coeff[12] + x + offsetZ + 4);
            neig3 = _mm_load_pd(coeff[12] + x + offsetZ + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[12] + x + offsetZ + offsetY + 0);
            neig1 = _mm_load_pd(coeff[12] + x + offsetZ + offsetY + 2);
            neig2 = _mm_load_pd(coeff[12] + x + offsetZ + offsetY + 4);
            neig3 = _mm_load_pd(coeff[12] + x + offsetZ + offsetY + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //yyyyyyyyyyyyy
            _mm_store_pd(dst + x + 0, same0);
            _mm_store_pd(dst + x + 2, same1);
            _mm_store_pd(dst + x + 4, same2);
            _mm_store_pd(dst + x + 6, same3);

            same0 = same4;
            neig0 = neig4;

            // dst[x] = 
            //     coeff[0][x] * src[x - offsetZ] +
            //     coeff[1][x] * src[x - offsetY] +
            //     coeff[2][x] * src[x - 1] +
            //     coeff[3][x] * src[x] +
            //     coeff[4][x] * src[x + 1] +
            //     coeff[5][x] * src[x + offsetY] +
            //     coeff[6][x] * src[x + offsetZ];
        }

        scalarUpdater.step(coeff, src, dst, offsetY, offsetZ, x, endX);
    }

    int flops()
    {
        return 40;
    }
};

template<int DIM_X, int DIM_Y, int DIM_Z>
class ExtendedVectorized3DFixed
{
public:
    static int coefficients()
    {
        return 13;
    }

    inline void step(double *coeff[13], double *src, double *dst, int unusedOffsetY, int unusedOffsetZ, int startX, int endX)
    {
        const int SLICE_SIZE = DIM_X * DIM_Y;
        const int TOTAL_SIZE = DIM_X * DIM_Y * DIM_Z;
        const int offsetY = DIM_X;
        const int offsetZ = SLICE_SIZE;

        int x = startX;
        ExtendedScalar3D scalarUpdater;

        if ((x & 1) == 1) {
            scalarUpdater.step(coeff, src, dst, DIM_Y, offsetZ, x, x + 1);
            x += 1;
        }

        __m128d same0 = _mm_load_pd(src + x + 0);
        __m128d neig0 = _mm_loadu_pd(src + x + 1);
        
        int paddedEndX = endX - 7;
        for (; x < paddedEndX; x += 8) {
            __m128d same1 = _mm_load_pd(src + x + 2);
            __m128d same2 = _mm_load_pd(src + x + 4);
            __m128d same3 = _mm_load_pd(src + x + 6);
            __m128d same4 = _mm_load_pd(src + x + 8);

            __m128d neig1 = _mm_shuffle_pd(same0, same1, (1 << 0) | (0 << 2));
            __m128d neig2 = _mm_shuffle_pd(same1, same2, (1 << 0) | (0 << 2));
            __m128d neig3 = _mm_shuffle_pd(same2, same3, (1 << 0) | (0 << 2));
            __m128d neig4 = _mm_shuffle_pd(same3, same4, (1 << 0) | (0 << 2));

            same0 = _mm_mul_pd(same0, _mm_load_pd(&coeff[0][TOTAL_SIZE * 3 + x + 0]));
            same1 = _mm_mul_pd(same1, _mm_load_pd(&coeff[0][TOTAL_SIZE * 3 + x + 2]));
            same2 = _mm_mul_pd(same2, _mm_load_pd(&coeff[0][TOTAL_SIZE * 3 + x + 4]));
            same3 = _mm_mul_pd(same3, _mm_load_pd(&coeff[0][TOTAL_SIZE * 3 + x + 6]));

            __m128d temp1 = _mm_mul_pd(neig0, _mm_load_pd(&coeff[0][TOTAL_SIZE * 2 + x + 0]));
            __m128d temp2 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[0][TOTAL_SIZE * 2 + x + 2]));
            __m128d temp3 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[0][TOTAL_SIZE * 2 + x + 4]));
            __m128d temp4 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[0][TOTAL_SIZE * 2 + x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            temp1 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[0][TOTAL_SIZE * 4 + x + 0]));
            temp2 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[0][TOTAL_SIZE * 4 + x + 2]));
            temp3 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[0][TOTAL_SIZE * 4 + x + 4]));
            temp4 = _mm_mul_pd(neig4, _mm_load_pd(&coeff[0][TOTAL_SIZE * 4 + x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            neig0 = _mm_load_pd(src + x - offsetZ + 0);
            neig1 = _mm_load_pd(src + x - offsetZ + 2);
            neig2 = _mm_load_pd(src + x - offsetZ + 4);
            neig3 = _mm_load_pd(src + x - offsetZ + 6);

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&coeff[0][TOTAL_SIZE * 0 + x + 0]));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[0][TOTAL_SIZE * 0 + x + 2]));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[0][TOTAL_SIZE * 0 + x + 4]));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[0][TOTAL_SIZE * 0 + x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            neig0 = _mm_load_pd(src + x - offsetY + 0);
            neig1 = _mm_load_pd(src + x - offsetY + 2);
            neig2 = _mm_load_pd(src + x - offsetY + 4);
            neig3 = _mm_load_pd(src + x - offsetY + 6);

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&coeff[0][TOTAL_SIZE * 1 + x + 0]));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[0][TOTAL_SIZE * 1 + x + 2]));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[0][TOTAL_SIZE * 1 + x + 4]));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[0][TOTAL_SIZE * 1 + x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            neig0 = _mm_load_pd(src + x + offsetY + 0);
            neig1 = _mm_load_pd(src + x + offsetY + 2);
            neig2 = _mm_load_pd(src + x + offsetY + 4);
            neig3 = _mm_load_pd(src + x + offsetY + 6);

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&coeff[0][TOTAL_SIZE * 5 + x + 0]));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[0][TOTAL_SIZE * 5 + x + 2]));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[0][TOTAL_SIZE * 5 + x + 4]));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[0][TOTAL_SIZE * 5 + x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(src + x + offsetZ + 0);
            neig1 = _mm_load_pd(src + x + offsetZ + 2);
            neig2 = _mm_load_pd(src + x + offsetZ + 4);
            neig3 = _mm_load_pd(src + x + offsetZ + 6);

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&coeff[0][TOTAL_SIZE * 6 + x + 0]));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[0][TOTAL_SIZE * 6 + x + 2]));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[0][TOTAL_SIZE * 6 + x + 4]));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[0][TOTAL_SIZE * 6 + x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(src + x - offsetZ - offsetY + 0);
            neig1 = _mm_load_pd(src + x - offsetZ - offsetY + 2);
            neig2 = _mm_load_pd(src + x - offsetZ - offsetY + 4);
            neig3 = _mm_load_pd(src + x - offsetZ - offsetY + 6);

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&coeff[0][TOTAL_SIZE * 7 + x + 0]));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[0][TOTAL_SIZE * 7 + x + 2]));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[0][TOTAL_SIZE * 7 + x + 4]));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[0][TOTAL_SIZE * 7 + x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(src + x - offsetZ + offsetY + 0);
            neig1 = _mm_load_pd(src + x - offsetZ + offsetY + 2);
            neig2 = _mm_load_pd(src + x - offsetZ + offsetY + 4);
            neig3 = _mm_load_pd(src + x - offsetZ + offsetY + 6);

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&coeff[0][TOTAL_SIZE * 8 + x + 0]));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[0][TOTAL_SIZE * 8 + x + 2]));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[0][TOTAL_SIZE * 8 + x + 4]));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[0][TOTAL_SIZE * 8 + x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(src + x + offsetZ - offsetY + 0);
            neig1 = _mm_load_pd(src + x + offsetZ - offsetY + 2);
            neig2 = _mm_load_pd(src + x + offsetZ - offsetY + 4);
            neig3 = _mm_load_pd(src + x + offsetZ - offsetY + 6);

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&coeff[0][TOTAL_SIZE * 9 + x + 0]));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[0][TOTAL_SIZE * 9 + x + 2]));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[0][TOTAL_SIZE * 9 + x + 4]));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[0][TOTAL_SIZE * 9 + x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(src + x + offsetZ + offsetY + 0);
            neig1 = _mm_load_pd(src + x + offsetZ + offsetY + 2);
            neig2 = _mm_load_pd(src + x + offsetZ + offsetY + 4);
            neig3 = _mm_load_pd(src + x + offsetZ + offsetY + 6);

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&coeff[0][TOTAL_SIZE * 10 + x + 0]));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&coeff[0][TOTAL_SIZE * 10 + x + 2]));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&coeff[0][TOTAL_SIZE * 10 + x + 4]));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&coeff[0][TOTAL_SIZE * 10 + x + 6]));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x - offsetZ - offsetY + 0);
            neig1 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x - offsetZ - offsetY + 2);
            neig2 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x - offsetZ - offsetY + 4);
            neig3 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x - offsetZ - offsetY + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x - offsetZ + 0);
            neig1 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x - offsetZ + 2);
            neig2 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x - offsetZ + 4);
            neig3 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x - offsetZ + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x - offsetZ + offsetY + 0);
            neig1 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x - offsetZ + offsetY + 2);
            neig2 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x - offsetZ + offsetY + 4);
            neig3 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x - offsetZ + offsetY + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x - offsetY + 0);
            neig1 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x - offsetY + 2);
            neig2 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x - offsetY + 4);
            neig3 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x - offsetY + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + offsetY + x + 0);
            neig1 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + offsetY + x + 2);
            neig2 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + offsetY + x + 4);
            neig3 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + offsetY + x + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x + 0);
            neig1 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x + 2);
            neig2 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x + 4);
            neig3 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x + offsetZ - offsetY + 0);
            neig1 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x + offsetZ - offsetY + 2);
            neig2 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x + offsetZ - offsetY + 4);
            neig3 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x + offsetZ - offsetY + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x + offsetZ + 0);
            neig1 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x + offsetZ + 2);
            neig2 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x + offsetZ + 4);
            neig3 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x + offsetZ + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x + offsetZ + offsetY + 0);
            neig1 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x + offsetZ + offsetY + 2);
            neig2 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x + offsetZ + offsetY + 4);
            neig3 = _mm_load_pd(coeff[0] + 11 * TOTAL_SIZE + x + offsetZ + offsetY + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x - offsetZ - offsetY + 0);
            neig1 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x - offsetZ - offsetY + 2);
            neig2 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x - offsetZ - offsetY + 4);
            neig3 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x - offsetZ - offsetY + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x - offsetZ + 0);
            neig1 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x - offsetZ + 2);
            neig2 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x - offsetZ + 4);
            neig3 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x - offsetZ + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x - offsetZ + offsetY + 0);
            neig1 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x - offsetZ + offsetY + 2);
            neig2 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x - offsetZ + offsetY + 4);
            neig3 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x - offsetZ + offsetY + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x - offsetY + 0);
            neig1 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x - offsetY + 2);
            neig2 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x - offsetY + 4);
            neig3 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x - offsetY + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + offsetY + x + 0);
            neig1 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + offsetY + x + 2);
            neig2 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + offsetY + x + 4);
            neig3 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + offsetY + x + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x + 0);
            neig1 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x + 2);
            neig2 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x + 4);
            neig3 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x + offsetZ - offsetY + 0);
            neig1 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x + offsetZ - offsetY + 2);
            neig2 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x + offsetZ - offsetY + 4);
            neig3 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x + offsetZ - offsetY + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x + offsetZ + 0);
            neig1 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x + offsetZ + 2);
            neig2 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x + offsetZ + 4);
            neig3 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x + offsetZ + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x + offsetZ + offsetY + 0);
            neig1 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x + offsetZ + offsetY + 2);
            neig2 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x + offsetZ + offsetY + 4);
            neig3 = _mm_load_pd(coeff[0] + 12 * TOTAL_SIZE + x + offsetZ + offsetY + 6);

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //yyyyyyyyyyyyy
            _mm_store_pd(dst + x + 0, same0);
            _mm_store_pd(dst + x + 2, same1);
            _mm_store_pd(dst + x + 4, same2);
            _mm_store_pd(dst + x + 6, same3);

            same0 = same4;
            neig0 = neig4;

            // dst[x] = 
            //     coeff[0][x] * src[x - offsetZ] +
            //     coeff[1][x] * src[x - offsetY] +
            //     coeff[2][x] * src[x - 1] +
            //     coeff[3][x] * src[x] +
            //     coeff[4][x] * src[x + 1] +
            //     coeff[5][x] * src[x + offsetY] +
            //     coeff[6][x] * src[x + offsetZ];
        }

        scalarUpdater.step(coeff, src, dst, offsetY, offsetZ, x, endX);
    }

    int flops()
    {
        return 40;
    }
};

template<int DIM_X, int DIM_Y, int DIM_Z>
class ExtendedVectorized3DNextGen
{
public:
    static int coefficients()
    {
        return 13;
    }

    template<class NEIGHBORHOOD>
    inline void step(const NEIGHBORHOOD& hood, double *dst, int unusedOffsetY, int unusedOffsetZ, int startX, int endX)
    {
        const int SLICE_SIZE = DIM_X * DIM_Y;
        const int TOTAL_SIZE = DIM_X * DIM_Y * DIM_Z;
        const int offsetY = DIM_X;
        const int offsetZ = SLICE_SIZE;

        int x = startX;

        if ((x & 1) == 1) {
            stepScalar(hood, dst, DIM_Y, offsetZ, x, x + 1);
            x += 1;
        }

        __m128d same0 = _mm_load_pd(&(hood.template src<0, 0, 0>(x)));
        __m128d neig0 = _mm_loadu_pd(&hood.template src<1, 0, 0>(x));

        int paddedEndX = endX - 7;
        for (; x < paddedEndX; x += 8) {
            __m128d same1 = _mm_load_pd(&hood.template src<2, 0, 0>(x));
            __m128d same2 = _mm_load_pd(&hood.template src<4, 0, 0>(x));
            __m128d same3 = _mm_load_pd(&hood.template src<6, 0, 0>(x));
            __m128d same4 = _mm_load_pd(&hood.template src<8, 0, 0>(x));

            __m128d neig1 = _mm_shuffle_pd(same0, same1, (1 << 0) | (0 << 2));
            __m128d neig2 = _mm_shuffle_pd(same1, same2, (1 << 0) | (0 << 2));
            __m128d neig3 = _mm_shuffle_pd(same2, same3, (1 << 0) | (0 << 2));
            __m128d neig4 = _mm_shuffle_pd(same3, same4, (1 << 0) | (0 << 2));

            same0 = _mm_mul_pd(same0, _mm_load_pd(&hood.template coeff<0, 0, 0, 3>(x)));
            same1 = _mm_mul_pd(same1, _mm_load_pd(&hood.template coeff<2, 0, 0, 3>(x)));
            same2 = _mm_mul_pd(same2, _mm_load_pd(&hood.template coeff<4, 0, 0, 3>(x)));
            same3 = _mm_mul_pd(same3, _mm_load_pd(&hood.template coeff<6, 0, 0, 3>(x)));

            __m128d temp1 = _mm_mul_pd(neig0, _mm_load_pd(&hood.template coeff<0, 0, 0, 2>(x)));
            __m128d temp2 = _mm_mul_pd(neig1, _mm_load_pd(&hood.template coeff<2, 0, 0, 2>(x)));
            __m128d temp3 = _mm_mul_pd(neig2, _mm_load_pd(&hood.template coeff<4, 0, 0, 2>(x)));
            __m128d temp4 = _mm_mul_pd(neig3, _mm_load_pd(&hood.template coeff<6, 0, 0, 2>(x)));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            temp1 = _mm_mul_pd(neig1, _mm_load_pd(&hood.template coeff<0, 0, 0, 4>(x)));
            temp2 = _mm_mul_pd(neig2, _mm_load_pd(&hood.template coeff<2, 0, 0, 4>(x)));
            temp3 = _mm_mul_pd(neig3, _mm_load_pd(&hood.template coeff<4, 0, 0, 4>(x)));
            temp4 = _mm_mul_pd(neig4, _mm_load_pd(&hood.template coeff<6, 0, 0, 4>(x)));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            neig0 = _mm_load_pd(&hood.template src<0, 0, -1>(x));
            neig1 = _mm_load_pd(&hood.template src<2, 0, -1>(x));
            neig2 = _mm_load_pd(&hood.template src<4, 0, -1>(x));
            neig3 = _mm_load_pd(&hood.template src<6, 0, -1>(x));

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&hood.template coeff<0, 0, 0, 0>(x)));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&hood.template coeff<2, 0, 0, 0>(x)));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&hood.template coeff<4, 0, 0, 0>(x)));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&hood.template coeff<6, 0, 0, 0>(x)));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            neig0 = _mm_load_pd(&hood.template src<0, -1, 0>(x));
            neig1 = _mm_load_pd(&hood.template src<2, -1, 0>(x));
            neig2 = _mm_load_pd(&hood.template src<4, -1, 0>(x));
            neig3 = _mm_load_pd(&hood.template src<6, -1, 0>(x));

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&hood.template coeff<0, 0, 0, 1>(x)));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&hood.template coeff<2, 0, 0, 1>(x)));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&hood.template coeff<4, 0, 0, 1>(x)));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&hood.template coeff<6, 0, 0, 1>(x)));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            neig0 = _mm_load_pd(&hood.template src<0, 1, 0>(x));
            neig1 = _mm_load_pd(&hood.template src<2, 1, 0>(x));
            neig2 = _mm_load_pd(&hood.template src<4, 1, 0>(x));
            neig3 = _mm_load_pd(&hood.template src<6, 1, 0>(x));

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&hood.template coeff<0, 0, 0, 5>(x)));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&hood.template coeff<2, 0, 0, 5>(x)));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&hood.template coeff<4, 0, 0, 5>(x)));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&hood.template coeff<6, 0, 0, 5>(x)));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood.template src<0, 0, 1>(x));
            neig1 = _mm_load_pd(&hood.template src<2, 0, 1>(x));
            neig2 = _mm_load_pd(&hood.template src<4, 0, 1>(x));
            neig3 = _mm_load_pd(&hood.template src<6, 0, 1>(x));

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&hood.template coeff<0, 0, 0, 6>(x)));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&hood.template coeff<2, 0, 0, 6>(x)));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&hood.template coeff<4, 0, 0, 6>(x)));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&hood.template coeff<6, 0, 0, 6>(x)));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood.template src<0, -1, -1>(x));
            neig1 = _mm_load_pd(&hood.template src<2, -1, -1>(x));
            neig2 = _mm_load_pd(&hood.template src<4, -1, -1>(x));
            neig3 = _mm_load_pd(&hood.template src<6, -1, -1>(x));

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&hood.template coeff<0, 0, 0, 7>(x)));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&hood.template coeff<2, 0, 0, 7>(x)));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&hood.template coeff<4, 0, 0, 7>(x)));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&hood.template coeff<6, 0, 0, 7>(x)));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood.template src<0, 1, -1>(x));
            neig1 = _mm_load_pd(&hood.template src<2, 1, -1>(x));
            neig2 = _mm_load_pd(&hood.template src<4, 1, -1>(x));
            neig3 = _mm_load_pd(&hood.template src<6, 1, -1>(x));

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&hood.template coeff<0, 0, 0, 8>(x)));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&hood.template coeff<2, 0, 0, 8>(x)));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&hood.template coeff<4, 0, 0, 8>(x)));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&hood.template coeff<6, 0, 0, 8>(x)));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood.template src<0, -1, 1>(x));
            neig1 = _mm_load_pd(&hood.template src<2, -1, 1>(x));
            neig2 = _mm_load_pd(&hood.template src<4, -1, 1>(x));
            neig3 = _mm_load_pd(&hood.template src<6, -1, 1>(x));

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&hood.template coeff<0, 0, 0, 9>(x)));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&hood.template coeff<2, 0, 0, 9>(x)));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&hood.template coeff<4, 0, 0, 9>(x)));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&hood.template coeff<6, 0, 0, 9>(x)));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood.template src<0, 1, 1>(x));
            neig1 = _mm_load_pd(&hood.template src<2, 1, 1>(x));
            neig2 = _mm_load_pd(&hood.template src<4, 1, 1>(x));
            neig3 = _mm_load_pd(&hood.template src<6, 1, 1>(x));

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&hood.template coeff<0, 0, 0, 10>(x)));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&hood.template coeff<2, 0, 0, 10>(x)));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&hood.template coeff<4, 0, 0, 10>(x)));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&hood.template coeff<6, 0, 0, 10>(x)));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood.template coeff<0, -1, -1, 11>(x));
            neig1 = _mm_load_pd(&hood.template coeff<2, -1, -1, 11>(x));
            neig2 = _mm_load_pd(&hood.template coeff<4, -1, -1, 11>(x));
            neig3 = _mm_load_pd(&hood.template coeff<6, -1, -1, 11>(x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood.template coeff<0, 0, -1, 11>(x));
            neig1 = _mm_load_pd(&hood.template coeff<2, 0, -1, 11>(x));
            neig2 = _mm_load_pd(&hood.template coeff<4, 0, -1, 11>(x));
            neig3 = _mm_load_pd(&hood.template coeff<6, 0, -1, 11>(x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood.template coeff<0, 1, -1, 11>(x));
            neig1 = _mm_load_pd(&hood.template coeff<2, 1, -1, 11>(x));
            neig2 = _mm_load_pd(&hood.template coeff<4, 1, -1, 11>(x));
            neig3 = _mm_load_pd(&hood.template coeff<6, 1, -1, 11>(x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood.template coeff<0, -1, 0, 11>(x));
            neig1 = _mm_load_pd(&hood.template coeff<2, -1, 0, 11>(x));
            neig2 = _mm_load_pd(&hood.template coeff<4, -1, 0, 11>(x));
            neig3 = _mm_load_pd(&hood.template coeff<6, -1, 0, 11>(x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood.template coeff<0, 1, 0, 11>(x));
            neig1 = _mm_load_pd(&hood.template coeff<2, 1, 0, 11>(x));
            neig2 = _mm_load_pd(&hood.template coeff<4, 1, 0, 11>(x));
            neig3 = _mm_load_pd(&hood.template coeff<6, 1, 0, 11>(x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood.template coeff<0, 0, 0, 11>(x));
            neig1 = _mm_load_pd(&hood.template coeff<2, 0, 0, 11>(x));
            neig2 = _mm_load_pd(&hood.template coeff<4, 0, 0, 11>(x));
            neig3 = _mm_load_pd(&hood.template coeff<6, 0, 0, 11>(x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood.template coeff<0, -1, 1, 11>(x));
            neig1 = _mm_load_pd(&hood.template coeff<2, -1, 1, 11>(x));
            neig2 = _mm_load_pd(&hood.template coeff<4, -1, 1, 11>(x));
            neig3 = _mm_load_pd(&hood.template coeff<6, -1, 1, 11>(x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood.template coeff<0, 0, 1, 11>(x));
            neig1 = _mm_load_pd(&hood.template coeff<2, 0, 1, 11>(x));
            neig2 = _mm_load_pd(&hood.template coeff<4, 0, 1, 11>(x));
            neig3 = _mm_load_pd(&hood.template coeff<6, 0, 1, 11>(x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood.template coeff<0, 1, 1, 11>(x));
            neig1 = _mm_load_pd(&hood.template coeff<2, 1, 1, 11>(x));
            neig2 = _mm_load_pd(&hood.template coeff<4, 1, 1, 11>(x));
            neig3 = _mm_load_pd(&hood.template coeff<6, 1, 1, 11>(x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood.template coeff<0, -1, -1, 12>(x));
            neig1 = _mm_load_pd(&hood.template coeff<2, -1, -1, 12>(x));
            neig2 = _mm_load_pd(&hood.template coeff<4, -1, -1, 12>(x));
            neig3 = _mm_load_pd(&hood.template coeff<6, -1, -1, 12>(x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood.template coeff<0, 0, -1, 12>(x));
            neig1 = _mm_load_pd(&hood.template coeff<2, 0, -1, 12>(x));
            neig2 = _mm_load_pd(&hood.template coeff<4, 0, -1, 12>(x));
            neig3 = _mm_load_pd(&hood.template coeff<6, 0, -1, 12>(x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood.template coeff<0, 1, -1, 12>(x));
            neig1 = _mm_load_pd(&hood.template coeff<2, 1, -1, 12>(x));
            neig2 = _mm_load_pd(&hood.template coeff<4, 1, -1, 12>(x));
            neig3 = _mm_load_pd(&hood.template coeff<6, 1, -1, 12>(x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood.template coeff<0, -1, 0, 12>(x));
            neig1 = _mm_load_pd(&hood.template coeff<2, -1, 0, 12>(x));
            neig2 = _mm_load_pd(&hood.template coeff<4, -1, 0, 12>(x));
            neig3 = _mm_load_pd(&hood.template coeff<6, -1, 0, 12>(x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood.template coeff<0, 1, 0, 12>(x));
            neig1 = _mm_load_pd(&hood.template coeff<2, 1, 0, 12>(x));
            neig2 = _mm_load_pd(&hood.template coeff<4, 1, 0, 12>(x));
            neig3 = _mm_load_pd(&hood.template coeff<6, 1, 0, 12>(x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood.template coeff<0, 0, 0, 12>(x));
            neig1 = _mm_load_pd(&hood.template coeff<2, 0, 0, 12>(x));
            neig2 = _mm_load_pd(&hood.template coeff<4, 0, 0, 12>(x));
            neig3 = _mm_load_pd(&hood.template coeff<6, 0, 0, 12>(x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood.template coeff<0, -1, 1, 12>(x));
            neig1 = _mm_load_pd(&hood.template coeff<2, -1, 1, 12>(x));
            neig2 = _mm_load_pd(&hood.template coeff<4, -1, 1, 12>(x));
            neig3 = _mm_load_pd(&hood.template coeff<6, -1, 1, 12>(x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood.template coeff<0, 0, 1, 12>(x));
            neig1 = _mm_load_pd(&hood.template coeff<2, 0, 1, 12>(x));
            neig2 = _mm_load_pd(&hood.template coeff<4, 0, 1, 12>(x));
            neig3 = _mm_load_pd(&hood.template coeff<6, 0, 1, 12>(x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood.template coeff<0, 1, 1, 12>(x));
            neig1 = _mm_load_pd(&hood.template coeff<2, 1, 1, 12>(x));
            neig2 = _mm_load_pd(&hood.template coeff<4, 1, 1, 12>(x));
            neig3 = _mm_load_pd(&hood.template coeff<6, 1, 1, 12>(x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //yyyyyyyyyyyyy
            _mm_store_pd(dst + x + 0, same0);
            _mm_store_pd(dst + x + 2, same1);
            _mm_store_pd(dst + x + 4, same2);
            _mm_store_pd(dst + x + 6, same3);

            same0 = same4;
            neig0 = neig4;

            // dst[x] = 
            //     coeff[0][x] * src[x - offsetZ] +
            //     coeff[1][x] * src[x - offsetY] +
            //     coeff[2][x] * src[x - 1] +
            //     coeff[3][x] * src[x] +
            //     coeff[4][x] * src[x + 1] +
            //     coeff[5][x] * src[x + offsetY] +
            //     coeff[6][x] * src[x + offsetZ];
        }

        stepScalar(hood, dst, offsetY, offsetZ, x, endX);
    }

    template<class NEIGHBORHOOD>
    inline void stepScalar(const NEIGHBORHOOD& hood, double *dst, int offsetY, int offsetZ, int startX, int endX)
    {
        for (int x = startX; x < endX; ++x) {
            dst[x] = 
                hood.template coeff<0, 0, 0, 0>(x) * hood.template src<0,    -1, -1>(x) +
                hood.template coeff<0, 0, 0, 1>(x) * hood.template src<0,     0, -1>(x) +
                hood.template coeff<0, 0, 0, 2>(x) * hood.template src<0,     1, -1>(x) +
                hood.template coeff<0, 0, 0, 3>(x) * hood.template src<0,    -1,  0>(x) + 
                hood.template coeff<0, 0, 0, 4>(x) * hood.template src<0 - 1, 0,  0>(x) +
                hood.template coeff<0, 0, 0, 5>(x) * hood.template src<0,     0,  0>(x) +
                hood.template coeff<0, 0, 0, 6>(x) * hood.template src<0 + 1, 0,  0>(x) +
                hood.template coeff<0, 0, 0, 7>(x) * hood.template src<0,     1,  0>(x) +
                hood.template coeff<0, 0, 0, 8>(x) * hood.template src<0,    -1,  1>(x) +
                hood.template coeff<0, 0, 0, 9>(x) * hood.template src<0,     0,  1>(x) +
                hood.template coeff<0, 0, 0,10>(x) * hood.template src<0,     1,  1>(x) +

                hood.template coeff<0, -1, -1, 11>(x) +
                hood.template coeff<0,  0, -1, 11>(x) +
                hood.template coeff<0,  1, -1, 11>(x) +
                hood.template coeff<0, -1,  0, 11>(x) +
                hood.template coeff<-1, 0,  0, 11>(x) +
                hood.template coeff<0,  0,  0, 11>(x) +
                hood.template coeff<1,  0,  0, 11>(x) +
                hood.template coeff<0,  1,  0, 11>(x) +
                hood.template coeff<0, -1,  1, 11>(x) +
                hood.template coeff<0,  0,  1, 11>(x) +
                hood.template coeff<0,  1,  1, 11>(x) +

                hood.template coeff<0, -1, -1, 12>(x) +
                hood.template coeff<0,  0, -1, 12>(x) +
                hood.template coeff<0,  1, -1, 12>(x) +
                hood.template coeff<0, -1,  0, 12>(x) +
                hood.template coeff<-1, 0,  0, 12>(x) +
                hood.template coeff<0,  0,  0, 12>(x) +
                hood.template coeff<1,  0,  0, 12>(x) +
                hood.template coeff<0,  1,  0, 12>(x) +
                hood.template coeff<0, -1,  1, 12>(x) +
                hood.template coeff<0,  0,  1, 12>(x) +
                hood.template coeff<0,  1,  1, 12>(x);
        }
    }

    int flops()
    {
        return 40;
    }
};

class ExtendedVectorized3DDefiant
{
public:
    static int coefficients()
    {
        return 13;
    }

    template<class NEIGHBORHOOD>
    inline void step(const NEIGHBORHOOD& hoody, double *dst, int startX, int endX)
    {
        int x = startX;

        if ((x & 1) == 1) {
            stepScalar(hoody, dst, x, x + 1);
            x += 1;
        }

#define hood(X, Y, Z) hoody[FixedCoord<X, Y, Z>()]
#define C00 FixedCoord< 0, 0, 0>()
#define C01 FixedCoord< 1, 0, 0>()
#define C02 FixedCoord< 2, 0, 0>()
#define C03 FixedCoord< 3, 0, 0>()
#define C04 FixedCoord< 4, 0, 0>()
#define C05 FixedCoord< 5, 0, 0>()
#define C06 FixedCoord< 6, 0, 0>()
#define C07 FixedCoord< 7, 0, 0>()
#define C08 FixedCoord< 8, 0, 0>()
#define C09 FixedCoord< 9, 0, 0>()
#define C10 FixedCoord<10, 0, 0>()
#define C11 FixedCoord<11, 0, 0>()
#define C12 FixedCoord<12, 0, 0>()

        __m128d same0 = _mm_load_pd(&(hood(0, 0, 0).src(x)));
        __m128d neig0 = _mm_loadu_pd(&(hood(1, 0, 0).src(x)));

        int paddedEndX = endX - 7;
        for (; x < paddedEndX; x += 8) {
            __m128d same1 = _mm_load_pd(&hood(2, 0, 0).src(x));
            __m128d same2 = _mm_load_pd(&hood(4, 0, 0).src(x));
            __m128d same3 = _mm_load_pd(&hood(6, 0, 0).src(x));
            __m128d same4 = _mm_load_pd(&hood(8, 0, 0).src(x));

            __m128d neig1 = _mm_shuffle_pd(same0, same1, (1 << 0) | (0 << 2));
            __m128d neig2 = _mm_shuffle_pd(same1, same2, (1 << 0) | (0 << 2));
            __m128d neig3 = _mm_shuffle_pd(same2, same3, (1 << 0) | (0 << 2));
            __m128d neig4 = _mm_shuffle_pd(same3, same4, (1 << 0) | (0 << 2));

            same0 = _mm_mul_pd(same0, _mm_load_pd(&hood(0, 0, 0).coeff(C03, x)));
            same1 = _mm_mul_pd(same1, _mm_load_pd(&hood(2, 0, 0).coeff(C03, x)));
            same2 = _mm_mul_pd(same2, _mm_load_pd(&hood(4, 0, 0).coeff(C03, x)));
            same3 = _mm_mul_pd(same3, _mm_load_pd(&hood(6, 0, 0).coeff(C03, x)));

            __m128d temp1 = _mm_mul_pd(neig0, _mm_load_pd(&hood(0, 0, 0).coeff(C02, x)));
            __m128d temp2 = _mm_mul_pd(neig1, _mm_load_pd(&hood(2, 0, 0).coeff(C02, x)));
            __m128d temp3 = _mm_mul_pd(neig2, _mm_load_pd(&hood(4, 0, 0).coeff(C02, x)));
            __m128d temp4 = _mm_mul_pd(neig3, _mm_load_pd(&hood(6, 0, 0).coeff(C02, x)));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            temp1 = _mm_mul_pd(neig1, _mm_load_pd(&hood(0, 0, 0).coeff(C04, x)));
            temp2 = _mm_mul_pd(neig2, _mm_load_pd(&hood(2, 0, 0).coeff(C04, x)));
            temp3 = _mm_mul_pd(neig3, _mm_load_pd(&hood(4, 0, 0).coeff(C04, x)));
            temp4 = _mm_mul_pd(neig4, _mm_load_pd(&hood(6, 0, 0).coeff(C04, x)));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            neig0 = _mm_load_pd(&hood(0, 0, -1).src(x));
            neig1 = _mm_load_pd(&hood(2, 0, -1).src(x));
            neig2 = _mm_load_pd(&hood(4, 0, -1).src(x));
            neig3 = _mm_load_pd(&hood(6, 0, -1).src(x));

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&hood(0, 0, 0).coeff(C00, x)));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&hood(2, 0, 0).coeff(C00, x)));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&hood(4, 0, 0).coeff(C00, x)));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&hood(6, 0, 0).coeff(C00, x)));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            neig0 = _mm_load_pd(&hood(0, -1, 0).src(x));
            neig1 = _mm_load_pd(&hood(2, -1, 0).src(x));
            neig2 = _mm_load_pd(&hood(4, -1, 0).src(x));
            neig3 = _mm_load_pd(&hood(6, -1, 0).src(x));

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&hood(0, 0, 0).coeff(C01, x)));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&hood(2, 0, 0).coeff(C01, x)));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&hood(4, 0, 0).coeff(C01, x)));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&hood(6, 0, 0).coeff(C01, x)));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            neig0 = _mm_load_pd(&hood(0, 1, 0).src(x));
            neig1 = _mm_load_pd(&hood(2, 1, 0).src(x));
            neig2 = _mm_load_pd(&hood(4, 1, 0).src(x));
            neig3 = _mm_load_pd(&hood(6, 1, 0).src(x));

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&hood(0, 0, 0).coeff(C05, x)));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&hood(2, 0, 0).coeff(C05, x)));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&hood(4, 0, 0).coeff(C05, x)));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&hood(6, 0, 0).coeff(C05, x)));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood(0, 0, 1).src(x));
            neig1 = _mm_load_pd(&hood(2, 0, 1).src(x));
            neig2 = _mm_load_pd(&hood(4, 0, 1).src(x));
            neig3 = _mm_load_pd(&hood(6, 0, 1).src(x));

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&hood(0, 0, 0).coeff(C06, x)));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&hood(2, 0, 0).coeff(C06, x)));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&hood(4, 0, 0).coeff(C06, x)));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&hood(6, 0, 0).coeff(C06, x)));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood(0, -1, -1).src(x));
            neig1 = _mm_load_pd(&hood(2, -1, -1).src(x));
            neig2 = _mm_load_pd(&hood(4, -1, -1).src(x));
            neig3 = _mm_load_pd(&hood(6, -1, -1).src(x));

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&hood(0, 0, 0).coeff(C07, x)));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&hood(2, 0, 0).coeff(C07, x)));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&hood(4, 0, 0).coeff(C07, x)));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&hood(6, 0, 0).coeff(C07, x)));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood(0, 1, -1).src(x));
            neig1 = _mm_load_pd(&hood(2, 1, -1).src(x));
            neig2 = _mm_load_pd(&hood(4, 1, -1).src(x));
            neig3 = _mm_load_pd(&hood(6, 1, -1).src(x));

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&hood(0, 0, 0).coeff(C08, x)));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&hood(2, 0, 0).coeff(C08, x)));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&hood(4, 0, 0).coeff(C08, x)));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&hood(6, 0, 0).coeff(C08, x)));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood(0, -1, 1).src(x));
            neig1 = _mm_load_pd(&hood(2, -1, 1).src(x));
            neig2 = _mm_load_pd(&hood(4, -1, 1).src(x));
            neig3 = _mm_load_pd(&hood(6, -1, 1).src(x));

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&hood(0, 0, 0).coeff(C09, x)));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&hood(2, 0, 0).coeff(C09, x)));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&hood(4, 0, 0).coeff(C09, x)));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&hood(6, 0, 0).coeff(C09, x)));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood(0, 1, 1).src(x));
            neig1 = _mm_load_pd(&hood(2, 1, 1).src(x));
            neig2 = _mm_load_pd(&hood(4, 1, 1).src(x));
            neig3 = _mm_load_pd(&hood(6, 1, 1).src(x));

            temp1 = _mm_mul_pd(neig0, _mm_load_pd(&hood(0, 0, 0).coeff(C10, x)));
            temp2 = _mm_mul_pd(neig1, _mm_load_pd(&hood(2, 0, 0).coeff(C10, x)));
            temp3 = _mm_mul_pd(neig2, _mm_load_pd(&hood(4, 0, 0).coeff(C10, x)));
            temp4 = _mm_mul_pd(neig3, _mm_load_pd(&hood(6, 0, 0).coeff(C10, x)));

            same0 = _mm_add_pd(same0, temp1);
            same1 = _mm_add_pd(same1, temp2);
            same2 = _mm_add_pd(same2, temp3);
            same3 = _mm_add_pd(same3, temp4);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood(0, -1, -1).coeff(C11, x));
            neig1 = _mm_load_pd(&hood(2, -1, -1).coeff(C11, x));
            neig2 = _mm_load_pd(&hood(4, -1, -1).coeff(C11, x));
            neig3 = _mm_load_pd(&hood(6, -1, -1).coeff(C11, x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood(0, 0, -1).coeff(C11, x));
            neig1 = _mm_load_pd(&hood(2, 0, -1).coeff(C11, x));
            neig2 = _mm_load_pd(&hood(4, 0, -1).coeff(C11, x));
            neig3 = _mm_load_pd(&hood(6, 0, -1).coeff(C11, x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood(0, 1, -1).coeff(C11, x));
            neig1 = _mm_load_pd(&hood(2, 1, -1).coeff(C11, x));
            neig2 = _mm_load_pd(&hood(4, 1, -1).coeff(C11, x));
            neig3 = _mm_load_pd(&hood(6, 1, -1).coeff(C11, x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood(0, -1, 0).coeff(C11, x));
            neig1 = _mm_load_pd(&hood(2, -1, 0).coeff(C11, x));
            neig2 = _mm_load_pd(&hood(4, -1, 0).coeff(C11, x));
            neig3 = _mm_load_pd(&hood(6, -1, 0).coeff(C11, x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood(0, 1, 0).coeff(C11, x));
            neig1 = _mm_load_pd(&hood(2, 1, 0).coeff(C11, x));
            neig2 = _mm_load_pd(&hood(4, 1, 0).coeff(C11, x));
            neig3 = _mm_load_pd(&hood(6, 1, 0).coeff(C11, x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood(0, 0, 0).coeff(C11, x));
            neig1 = _mm_load_pd(&hood(2, 0, 0).coeff(C11, x));
            neig2 = _mm_load_pd(&hood(4, 0, 0).coeff(C11, x));
            neig3 = _mm_load_pd(&hood(6, 0, 0).coeff(C11, x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood(0, -1, 1).coeff(C11, x));
            neig1 = _mm_load_pd(&hood(2, -1, 1).coeff(C11, x));
            neig2 = _mm_load_pd(&hood(4, -1, 1).coeff(C11, x));
            neig3 = _mm_load_pd(&hood(6, -1, 1).coeff(C11, x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood(0, 0, 1).coeff(C11, x));
            neig1 = _mm_load_pd(&hood(2, 0, 1).coeff(C11, x));
            neig2 = _mm_load_pd(&hood(4, 0, 1).coeff(C11, x));
            neig3 = _mm_load_pd(&hood(6, 0, 1).coeff(C11, x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood(0, 1, 1).coeff(C11, x));
            neig1 = _mm_load_pd(&hood(2, 1, 1).coeff(C11, x));
            neig2 = _mm_load_pd(&hood(4, 1, 1).coeff(C11, x));
            neig3 = _mm_load_pd(&hood(6, 1, 1).coeff(C11, x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood(0, -1, -1).coeff(C12, x));
            neig1 = _mm_load_pd(&hood(2, -1, -1).coeff(C12, x));
            neig2 = _mm_load_pd(&hood(4, -1, -1).coeff(C12, x));
            neig3 = _mm_load_pd(&hood(6, -1, -1).coeff(C12, x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood(0, 0, -1).coeff(C12, x));
            neig1 = _mm_load_pd(&hood(2, 0, -1).coeff(C12, x));
            neig2 = _mm_load_pd(&hood(4, 0, -1).coeff(C12, x));
            neig3 = _mm_load_pd(&hood(6, 0, -1).coeff(C12, x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood(0, 1, -1).coeff(C12, x));
            neig1 = _mm_load_pd(&hood(2, 1, -1).coeff(C12, x));
            neig2 = _mm_load_pd(&hood(4, 1, -1).coeff(C12, x));
            neig3 = _mm_load_pd(&hood(6, 1, -1).coeff(C12, x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood(0, -1, 0).coeff(C12, x));
            neig1 = _mm_load_pd(&hood(2, -1, 0).coeff(C12, x));
            neig2 = _mm_load_pd(&hood(4, -1, 0).coeff(C12, x));
            neig3 = _mm_load_pd(&hood(6, -1, 0).coeff(C12, x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood(0, 1, 0).coeff(C12, x));
            neig1 = _mm_load_pd(&hood(2, 1, 0).coeff(C12, x));
            neig2 = _mm_load_pd(&hood(4, 1, 0).coeff(C12, x));
            neig3 = _mm_load_pd(&hood(6, 1, 0).coeff(C12, x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood(0, 0, 0).coeff(C12, x));
            neig1 = _mm_load_pd(&hood(2, 0, 0).coeff(C12, x));
            neig2 = _mm_load_pd(&hood(4, 0, 0).coeff(C12, x));
            neig3 = _mm_load_pd(&hood(6, 0, 0).coeff(C12, x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood(0, -1, 1).coeff(C12, x));
            neig1 = _mm_load_pd(&hood(2, -1, 1).coeff(C12, x));
            neig2 = _mm_load_pd(&hood(4, -1, 1).coeff(C12, x));
            neig3 = _mm_load_pd(&hood(6, -1, 1).coeff(C12, x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood(0, 0, 1).coeff(C12, x));
            neig1 = _mm_load_pd(&hood(2, 0, 1).coeff(C12, x));
            neig2 = _mm_load_pd(&hood(4, 0, 1).coeff(C12, x));
            neig3 = _mm_load_pd(&hood(6, 0, 1).coeff(C12, x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //xxxxxxxxxxxxx
            neig0 = _mm_load_pd(&hood(0, 1, 1).coeff(C12, x));
            neig1 = _mm_load_pd(&hood(2, 1, 1).coeff(C12, x));
            neig2 = _mm_load_pd(&hood(4, 1, 1).coeff(C12, x));
            neig3 = _mm_load_pd(&hood(6, 1, 1).coeff(C12, x));

            same0 = _mm_add_pd(same0, neig0);
            same1 = _mm_add_pd(same1, neig1);
            same2 = _mm_add_pd(same2, neig2);
            same3 = _mm_add_pd(same3, neig3);

            //yyyyyyyyyyyyy
            _mm_store_pd(dst + x + 0, same0);
            _mm_store_pd(dst + x + 2, same1);
            _mm_store_pd(dst + x + 4, same2);
            _mm_store_pd(dst + x + 6, same3);

            same0 = same4;
            neig0 = neig4;

            // dst[x] = 
            //     coeff[0][x] * src[x - offsetZ] +
            //     coeff[1][x] * src[x - offsetY] +
            //     coeff[2][x] * src[x - 1] +
            //     coeff[3][x] * src[x] +
            //     coeff[4][x] * src[x + 1] +
            //     coeff[5][x] * src[x + offsetY] +
            //     coeff[6][x] * src[x + offsetZ];
        }

        stepScalar(hoody, dst, x, endX);
    }

    template<class NEIGHBORHOOD>
    inline void stepScalar(const NEIGHBORHOOD& hoody, double *dst, int startX, int endX)
    {
        for (int x = startX; x < endX; ++x) {
            dst[x] = 
                hood(0, 0, 0).coeff(C00, x) * hood( 0, -1, -1).src(x) +
                hood(0, 0, 0).coeff(C01, x) * hood( 0,  0, -1).src(x) +
                hood(0, 0, 0).coeff(C02, x) * hood( 0,  1, -1).src(x) +
                hood(0, 0, 0).coeff(C03, x) * hood( 0, -1,  0).src(x) +
                hood(0, 0, 0).coeff(C04, x) * hood(-1,  0,  0).src(x) +
                hood(0, 0, 0).coeff(C05, x) * hood( 0,  0,  0).src(x) +
                hood(0, 0, 0).coeff(C06, x) * hood( 1,  0,  0).src(x) +
                hood(0, 0, 0).coeff(C07, x) * hood( 0,  1,  0).src(x) +
                hood(0, 0, 0).coeff(C08, x) * hood( 0, -1,  1).src(x) +
                hood(0, 0, 0).coeff(C09, x) * hood( 0,  0,  1).src(x) +
                hood(0, 0, 0).coeff(C10, x) * hood( 0,  1,  1).src(x) +

                hood(0, -1, -1).coeff(C11, x) +
                hood(0,  0, -1).coeff(C11, x) +
                hood(0,  1, -1).coeff(C11, x) +
                hood(0, -1,  0).coeff(C11, x) +
                hood(-1, 0,  0).coeff(C11, x) +
                hood(0,  0,  0).coeff(C11, x) +
                hood(1,  0,  0).coeff(C11, x) +
                hood(0,  1,  0).coeff(C11, x) +
                hood(0, -1,  1).coeff(C11, x) +
                hood(0,  0,  1).coeff(C11, x) +
                hood(0,  1,  1).coeff(C11, x) +

                hood(0, -1, -1).coeff(C12, x) +
                hood(0,  0, -1).coeff(C12, x) +
                hood(0,  1, -1).coeff(C12, x) +
                hood(0, -1,  0).coeff(C12, x) +
                hood(-1, 0,  0).coeff(C12, x) +
                hood(0,  0,  0).coeff(C12, x) +
                hood(1,  0,  0).coeff(C12, x) +
                hood(0,  1,  0).coeff(C12, x) +
                hood(0, -1,  1).coeff(C12, x) +
                hood(0,  0,  1).coeff(C12, x) +
                hood(0,  1,  1).coeff(C12, x);
        }
    }

    int flops()
    {
        return 40;
    }
};

int main(int argc, char *argv[])
{
    // Benchmark<Scalar3D, 3>().exercise();
    // Benchmark<Vectorized3D, 3>().exercise();
    // Benchmark<VectorizedSSEMelbourneShuffle2D>().exercise();
    // Benchmark<ExtendedVectorized3D, 3>().exercise();
    // Benchmark<ExtendedVectorized3DFixed, 3>().exercise();
    // Benchmark<ExtendedVectorized3DNextGen<4, 4, MY_SIZE>, 3>().exercise();
    Benchmark<ExtendedVectorized3DDefiant, 3>().exercise();
    // Benchmark<Jacobi3D, 3>().exercise();


    // std::vector<cl::Platform> platforms;
    // cl::Platform::get(&platforms);

    // for (int i = 0; i < platforms.size(); ++i) {
    //     std::string str;
    //     platforms[i].getInfo(CL_PLATFORM_NAME, &str);
    //     std::cout << "Platform[" << i << "] = " << str << std::endl;
    // }

    return 0;
}
