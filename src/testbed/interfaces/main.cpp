#include <cmath>
#include <typeinfo> 
// #include <CL/cl.h>
#include <iostream>
// #include <pmmintrin.h>
#include <sys/time.h>
#include <vector>

#include "interfaces.h"

class VanillaRunner
{
public:
    VanillaRunner(Coord<2> _dim) :
        dim(_dim),
        gridA(dim),
        gridB(dim)
    {}

    double run(long repeats)
    {
        double *oldGrid = &gridA[Coord<2>(0, 0)];
        double *newGrid = &gridB[Coord<2>(0, 0)];
        
        for (long t = 0; t < repeats; ++t) {
            for (int y = 1; y < dim.y() - 1; ++y) {
                for (int x = 1; x < dim.x() - 1; ++x) {
                    newGrid[y * dim.y() + x] = 
                        (oldGrid[(y - 1) * dim.y() + x + 0] +
                         oldGrid[(y + 0) * dim.y() + x - 1] + 
                         oldGrid[(y + 0) * dim.y() + x + 0] + 
                         oldGrid[(y + 0) * dim.y() + x + 1] + 
                         oldGrid[(y + 1) * dim.y() + x + 0]) * (1.0 / 5.0);
                }
            }

            std::swap(oldGrid, newGrid);
        }

        return gridA[Coord<2>(1, 1)];
    }

    static int flops()
    {
        return 5;
    }

private:
    Coord<2> dim;
    GridType gridA;
    GridType gridB;
};

class VirtualInterfaceRunner
{
public:
    VirtualInterfaceRunner(Coord<2> _dim) :
        dim(_dim),
        gridA(dim),
        gridB(dim)
    {}

    double run(long repeats)
    {
        VirtualGridType *oldGrid = &gridA;
        VirtualGridType *newGrid = &gridB;
        
        for (long t = 0; t < repeats; ++t) {
            for (int y = 1; y < dim.y() - 1; ++y) {
                for (int x = 1; x < dim.x() - 1; ++x) {
                    CellInterface& c = (*newGrid)[Coord<2>(x, y)];
                    HoodOld hood(oldGrid, Coord<2>(x, y));
                    c.update(hood);
                }
            }

            std::swap(oldGrid, newGrid);
        }

        return gridA[Coord<2>(1, 1)].getVal();
    }

    static int flops()
    {
        return 5;
    }

private:
    Coord<2> dim;
    VirtualGridType gridA;
    VirtualGridType gridB;
};

class StraightInterfaceRunner
{
public:
    StraightInterfaceRunner(Coord<2> _dim) :
        dim(_dim),
        gridA(dim),
        gridB(dim)
    {}

    double run(long repeats)
    {
        StraightGridType *oldGrid = &gridA;
        StraightGridType *newGrid = &gridB;
        
        for (long t = 0; t < repeats; ++t) {
            for (int y = 1; y < dim.y() - 1; ++y) {
                for (int x = 1; x < dim.x() - 1; ++x) {
                    CellStraight *c = &(*newGrid)[Coord<2>(x, y)];
                    HoodStraight hood(oldGrid, Coord<2>(x, y));
                    c->update(hood);
                }
            }

            std::swap(oldGrid, newGrid);
        }

        return gridA[Coord<2>(1, 1)].val;
    }

    static int flops()
    {
        return 5;
    }

private:
    Coord<2> dim;
    StraightGridType gridA;
    StraightGridType gridB;
};

class StreakInterfaceRunner
{
public:
    StreakInterfaceRunner(Coord<2> _dim) :
        dim(_dim),
        gridA(dim),
        gridB(dim)
    {}

    double run(long repeats)
    {
        StraightGridType *oldGrid = &gridA;
        StraightGridType *newGrid = &gridB;
        
        for (long t = 0; t < repeats; ++t) {
            for (int y = 1; y < dim.y() - 1; ++y) {
                CellStraight *c = &(*oldGrid)[Coord<2>(0, y)];
                CellStraight *lines[3] = {
                    &(*oldGrid)[Coord<2>(0, y - 1)],
                    &(*oldGrid)[Coord<2>(0, y + 0)],
                    &(*oldGrid)[Coord<2>(0, y + 1)]
                };
                HoodStreak hood(lines);
                
                CellStraight::updateStreak(c, hood, 1, dim.x() - 1);
            }

            std::swap(oldGrid, newGrid);
        }

        return gridA[Coord<2>(1, 1)].val;
    }

    static int flops()
    {
        return 5;
    }

private:
    Coord<2> dim;
    StraightGridType gridA;
    StraightGridType gridB;
};

class StreakInterfaceRunnerSSE
{
public:
    StreakInterfaceRunnerSSE(Coord<2> _dim) :
        dim(_dim),
        gridA(dim),
        gridB(dim)
    {}

    double run(long repeats)
    {
        StraightGridType *oldGrid = &gridA;
        StraightGridType *newGrid = &gridB;
        
        for (long t = 0; t < repeats; ++t) {
            for (int y = 1; y < dim.y() - 1; ++y) {
                CellStraight *c = &(*oldGrid)[Coord<2>(0, y)];
                CellStraight *lines[3] = {
                    &(*oldGrid)[Coord<2>(0, y - 1)],
                    &(*oldGrid)[Coord<2>(0, y + 0)],
                    &(*oldGrid)[Coord<2>(0, y + 1)]
                };
                HoodStreak hood(lines);
                
                CellStraight::updateStreakSSE(c, hood, 1, dim.x() - 1);
            }

            std::swap(oldGrid, newGrid);
        }

        return gridA[Coord<2>(1, 1)].val;
    }

    static int flops()
    {
        return 5;
    }

private:
    Coord<2> dim;
    StraightGridType gridA;
    StraightGridType gridB;
};


template<typename RUNNER, int DIM=2>
class Benchmark
{
public:
    void run(Coord<DIM> dim, int repeats)
    {
        RUNNER r(dim);

        long long tStart = getUTtime();
        r.run(repeats);
        long long tEnd = getUTtime();
        evaluate(dim, repeats, tEnd - tStart);
    }

    void exercise() 
    {
        std::cout << "# " << typeid(RUNNER).name() << "\n";
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
                    dim[0] = d;
                    dim[1] = 4;
                    dim[2] = 4;
                    // for (int i = 0; i < DIM; ++i)
                    //     dim[i] = d;
                    int repeats = std::max(1, 100000000 / dim.prod());
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
        Coord<DIM> inner = dim - Coord<DIM>::diagonal(2);
        double gflops = 1.0 * RUNNER::flops() * inner.prod() * 
            repeats / 1000 / 1000 / 1000 / seconds;
        std::cout << dim.x() << " " << gflops << "\n";
    }
};


int main(int argc, char *argv[])
{
    // Benchmark<VanillaRunner>().exercise();
    // Benchmark<VirtualInterfaceRunner>().exercise();
    // Benchmark<StraightInterfaceRunner>().exercise();
    // Benchmark<StreakInterfaceRunner>().exercise();
    Benchmark<StreakInterfaceRunnerSSE>().exercise();
    return 0;
}
