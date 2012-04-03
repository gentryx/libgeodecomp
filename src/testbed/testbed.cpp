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

typedef Grid<double> GridType;

class Scalar
{
public:
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

class VectorizedSSEMelbourneShuffle
{
public:
    inline void step(double *src, double *dst, int offset, int startX, int endX)
    {
        int x = startX;
        Scalar scalarUpdater;

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

class VectorizedSSEMelbourneShuffleB
{
public:
    inline void step(double *src, double *dst, int offset, int startX, int endX)
    {
        int x = startX;
        Scalar scalarUpdater;

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
    
            // load top row
            __m128d temp0 = _mm_load_pd(src - offset + x + 0);
            __m128d temp1 = _mm_load_pd(src - offset + x + 2);
            __m128d temp2 = _mm_load_pd(src - offset + x + 4);
            __m128d temp3 = _mm_load_pd(src - offset + x + 6);
            // add top row
            same0 = _mm_add_pd(same0, temp0);
            same1 = _mm_add_pd(same1, temp1);
            same2 = _mm_add_pd(same2, temp2);
            same3 = _mm_add_pd(same3, temp3);

            // load bottom row
            buff0 = _mm_load_pd(src + offset + x + 0);
            buff1 = _mm_load_pd(src + offset + x + 2);
            buff2 = _mm_load_pd(src + offset + x + 4);
            buff3 = _mm_load_pd(src + offset + x + 6);
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

class VectorizedSSEMelbourneShuffleC
{
public:
    inline void step(double *src, double *dst, int offset, int startX, int endX)
    {
        int x = startX;
        Scalar scalarUpdater;

        if ((x & 1) == 1) {
            scalarUpdater.step(src, dst, offset, x, x + 1);
            x += 1;
        }

        __m128d oneFifth = _mm_set_pd(1.0/3.0, 1.0/3.0);
        __m128d buff0 = _mm_loadu_pd(src + x - 1);
        __m128d same0 = _mm_load_pd(src + x + 0);

        int paddedEndX = endX - 7;
        for (; x < paddedEndX; x += 8) {

            __m128d same1 = _mm_load_pd(src + x + 2);
            __m128d buff1 = _mm_shuffle_pd(same0, same1, (1 << 0) | (0 << 2));
            same0 = _mm_add_pd(same0, buff0);
            same0 = _mm_add_pd(same0, buff1);
            __m128d temp0 = _mm_load_pd(src - offset + x + 0);
            same0 = _mm_add_pd(same0, temp0);
            temp0 = _mm_load_pd(src + offset + x + 0);
            same0 = _mm_add_pd(same0, temp0);
            same0 = _mm_mul_pd(same0, oneFifth);
            _mm_store_pd(dst + 0, same0);

            __m128d same2 = _mm_load_pd(src + x + 4);
            __m128d buff2 = _mm_shuffle_pd(same1, same2, (1 << 0) | (0 << 2));
            same1 = _mm_add_pd(same1, buff1);
            same1 = _mm_add_pd(same1, buff2);
            __m128d temp1 = _mm_load_pd(src - offset + x + 2);
            same1 = _mm_add_pd(same1, temp1);
            temp1 = _mm_load_pd(src + offset + x + 2);
            same1 = _mm_add_pd(same1, temp1);
            same1 = _mm_mul_pd(same1, oneFifth);
            _mm_store_pd(dst + 2, same1);

            __m128d same3 = _mm_load_pd(src + x + 6);
            __m128d buff3 = _mm_shuffle_pd(same2, same3, (1 << 0) | (0 << 2));
            same2 = _mm_add_pd(same2, buff2);
            same2 = _mm_add_pd(same2, buff3);
            __m128d temp2 = _mm_load_pd(src - offset + x + 4);
            same2 = _mm_add_pd(same2, temp2);
            temp2 = _mm_load_pd(src + offset + x + 4);
            same2 = _mm_add_pd(same2, temp2);
            same2 = _mm_mul_pd(same2, oneFifth);
            _mm_store_pd(dst + 4, same2);

            __m128d same4 = _mm_load_pd(src + x + 8);
            __m128d buff4 = _mm_shuffle_pd(same3, same4, (1 << 0) | (0 << 2));
            same3 = _mm_add_pd(same3, buff3);
            same3 = _mm_add_pd(same3, buff4);
            __m128d temp3 = _mm_load_pd(src - offset + x + 6);
            same3 = _mm_add_pd(same3, temp3);
            temp3 = _mm_load_pd(src + offset + x + 6);
            same3 = _mm_add_pd(same3, temp3);
            same3 = _mm_mul_pd(same3, oneFifth);
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


template<typename UPDATER>
class Benchmark
{
public:
    void run(Coord<2> dim, int repeats)
    {
        GridType a(dim);
        GridType b(dim);

        GridType *oldGrid = &a;
        GridType *newGrid = &b;

        int height = dim.y();
        int width = dim.x();

        UPDATER updater;

        long long tStart = getUTtime();

        for (int t = 0; t < repeats; ++t) {
            for (int y = 1; y < height - 1; ++y) {
                Coord<2> c(0, y);
                updater.step(&oldGrid->at(c), &newGrid->at(c), width, 1, width - 1);
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
                    Coord<2> dim(d, d);
                    int repeats = std::max(1, 500000000 / dim.prod());
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

    void evaluate(Coord<2> dim, int repeats, long long uTime)
    {
        double seconds = 1.0 * uTime / 1000 / 1000;
        double gflops = 1.0 * UPDATER().flops() * (dim.x() - 2) * (dim.y() - 2) * 
            repeats / 1000 / 1000 / 1000 / seconds;
        std::cout << dim.x() << " " << dim.y() << " " << gflops << "\n";
    }


};

int main(int argc, char *argv[])
{
    Benchmark<Scalar>().exercise();
    Benchmark<VectorizedSSEMelbourneShuffle>().exercise();
    Benchmark<VectorizedSSEMelbourneShuffleB>().exercise();
    Benchmark<VectorizedSSEMelbourneShuffleC>().exercise();


    // std::vector<cl::Platform> platforms;
    // cl::Platform::get(&platforms);

    // for (int i = 0; i < platforms.size(); ++i) {
    //     std::string str;
    //     platforms[i].getInfo(CL_PLATFORM_NAME, &str);
    //     std::cout << "Platform[" << i << "] = " << str << std::endl;
    // }

    return 0;
}
