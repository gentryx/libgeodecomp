#include <libgeodecomp/config.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/geometry/coord.h>
#include <libgeodecomp/geometry/floatcoord.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/geometry/stencils.h>
#include <libgeodecomp/geometry/partitions/hindexingpartition.h>
#include <libgeodecomp/geometry/partitions/hilbertpartition.h>
#include <libgeodecomp/geometry/partitions/stripingpartition.h>
#include <libgeodecomp/geometry/partitions/zcurvepartition.h>
#include <libgeodecomp/storage/grid.h>
#include <libgeodecomp/storage/linepointerassembly.h>
#include <libgeodecomp/storage/linepointerupdatefunctor.h>
#include <libgeodecomp/storage/updatefunctor.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/testbed/performancetests/cpubenchmark.h>

#include <libflatarray/short_vec.hpp>
#include <libflatarray/testbed/cpu_benchmark.hpp>
#include <libflatarray/testbed/evaluate.hpp>

#include <emmintrin.h>
#ifdef __AVX__
#include <immintrin.h>
#endif

#include <iomanip>
#include <iostream>
#include <stdio.h>

using namespace LibGeoDecomp;

class RegionCount : public CPUBenchmark
{
public:
    std::string family()
    {
        return "RegionCount";
    }

    std::string species()
    {
        return "gold";
    }

    double performance2(const Coord<3>& dim)
    {
        int sum = 0;
        Region<3> r;
        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                r << Streak<3>(Coord<3>(0, y, z), dim.x());
            }
        }

        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            for (int z = 0; z < dim.z(); z += 4) {
                for (int y = 0; y < dim.y(); y += 4) {
                    for (int x = 0; x < dim.x(); x += 4) {
                        sum += r.count(Coord<3>(x, y, z));
                    }
                }
            }
        }

        if (sum == 31) {
            std::cout << "pure debug statement to prevent the compiler from optimizing away the previous loop";
        }

        return seconds;
    }

    std::string unit()
    {
        return "s";
    }
};

class RegionInsert : public CPUBenchmark
{
public:
    std::string family()
    {
        return "RegionInsert";
    }

    std::string species()
    {
        return "gold";
    }

    double performance2(const Coord<3>& dim)
    {
        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            Region<3> r;
            for (int z = 0; z < dim.z(); ++z) {
                for (int y = 0; y < dim.y(); ++y) {
                    r << Streak<3>(Coord<3>(0, y, z), dim.x());
                }
            }
        }

        return seconds;
    }

    std::string unit()
    {
        return "s";
    }
};

class RegionIntersect : public CPUBenchmark
{
public:
    std::string family()
    {
        return "RegionIntersect";
    }

    std::string species()
    {
        return "gold";
    }

    double performance2(const Coord<3>& dim)
    {
        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            Region<3> r1;
            Region<3> r2;

            for (int z = 0; z < dim.z(); ++z) {
                for (int y = 0; y < dim.y(); ++y) {
                    r1 << Streak<3>(Coord<3>(0, y, z), dim.x());
                }
            }

            for (int z = 1; z < (dim.z() - 1); ++z) {
                for (int y = 1; y < (dim.y() - 1); ++y) {
                    r2 << Streak<3>(Coord<3>(1, y, z), dim.x() - 1);
                }
            }

            Region<3> r3 = r1 & r2;
        }

        return seconds;
    }

    std::string unit()
    {
        return "s";
    }
};

class CoordEnumerationVanilla : public CPUBenchmark
{
public:
    std::string family()
    {
        return "CoordEnumeration";
    }

    std::string species()
    {
        return "vanilla";
    }

    double performance2(const Coord<3>& dim)
    {
        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            Coord<3> sum;

            for (int z = 0; z < dim.z(); ++z) {
                for (int y = 0; y < dim.y(); ++y) {
                    for (int x = 0; x < dim.x(); ++x) {
                        sum += Coord<3>(x, y, z);
                    }
                }
            }
            // trick the compiler to not optimize away the loop above
            if (sum == Coord<3>(1, 2, 3)) {
                std::cout << "whatever";
            }

        }

        return seconds;
    }

    std::string unit()
    {
        return "s";
    }
};

class CoordEnumerationBronze : public CPUBenchmark
{
public:
    std::string family()
    {
        return "CoordEnumeration";
    }

    std::string species()
    {
        return "bronze";
    }

    double performance2(const Coord<3>& dim)
    {
        Region<3> region;
        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                region << Streak<3>(Coord<3>(0, y, z), dim.x());
            }
        }

        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            Coord<3> sum;
            for (Region<3>::Iterator i = region.begin(); i != region.end(); ++i) {
                sum += *i;
            }

            // trick the compiler to not optimize away the loop above
            if (sum == Coord<3>(1, 2, 3)) {
                std::cout << "whatever";
            }
        }

        return seconds;
    }

    std::string unit()
    {
        return "s";
    }
};

class CoordEnumerationGold : public CPUBenchmark
{
public:
    std::string family()
    {
        return "CoordEnumeration";
    }

    std::string species()
    {
        return "gold";
    }

    double performance2(const Coord<3>& dim)
    {
        Region<3> region;
        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                region << Streak<3>(Coord<3>(0, y, z), dim.x());
            }
        }

        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            Coord<3> sum;
            for (Region<3>::StreakIterator i = region.beginStreak(); i != region.endStreak(); ++i) {
                Coord<3> c = i->origin;
                for (; c.x() < i->endX; c.x() += 1) {
                    sum += c;
                }
            }
            // trick the compiler to not optimize away the loop above
            if (sum == Coord<3>(1, 2, 3)) {
                std::cout << "whatever";
            }
        }

        return seconds;
    }

    std::string unit()
    {
        return "s";
    }
};

class FloatCoordAccumulationGold : public CPUBenchmark
{
public:
    std::string family()
    {
        return "FloatCoordAccumulation";
    }

    std::string species()
    {
        return "gold";
    }

    double performance2(const Coord<3>& dim)
    {
        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            FloatCoord<3> sum;

            for (int z = 0; z < dim.z(); ++z) {
                for (int y = 0; y < dim.y(); ++y) {
                    for (int x = 0; x < dim.x(); ++x) {
                        FloatCoord<3> addent(x, y, z);
                        sum += addent;
                    }
                }
            }

            // trick the compiler to not optimize away the loop above
            if (sum == FloatCoord<3>(1, 2, 3)) {
                std::cout << "whatever";
            }

        }

        return seconds;
    }

    std::string unit()
    {
        return "s";
    }
};

class Jacobi3DVanilla : public CPUBenchmark
{
public:
    std::string family()
    {
        return "Jacobi3D";
    }

    std::string species()
    {
        return "vanilla";
    }

    double performance2(const Coord<3>& dim)
    {
        int dimX = dim.x();
        int dimY = dim.y();
        int dimZ = dim.z();
        int offsetZ = dimX * dimY;
        int maxT = 20;

        double *gridOld = new double[dimX * dimY * dimZ];
        double *gridNew = new double[dimX * dimY * dimZ];

        for (int z = 0; z < dimZ; ++z) {
            for (int y = 0; y < dimY; ++y) {
                for (int x = 0; x < dimY; ++x) {
                    gridOld[z * offsetZ + y * dimY + x] = x + y + z;
                    gridNew[z * offsetZ + y * dimY + x] = x + y + z;
                }
            }
        }

        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            for (int t = 0; t < maxT; ++t) {
                for (int z = 1; z < (dimZ - 1); ++z) {
                    for (int y = 1; y < (dimY - 1); ++y) {
                        for (int x = 1; x < (dimX - 1); ++x) {
                            gridNew[z * offsetZ + y * dimY + x] =
                                (gridOld[z * offsetZ + y * dimX + x - offsetZ] +
                                 gridOld[z * offsetZ + y * dimX + x - dimX] +
                                 gridOld[z * offsetZ + y * dimX + x - 1] +
                                 gridOld[z * offsetZ + y * dimX + x + 0] +
                                 gridOld[z * offsetZ + y * dimX + x + 1] +
                                 gridOld[z * offsetZ + y * dimX + x + dimX] +
                                 gridOld[z * offsetZ + y * dimX + x + offsetZ]) * (1.0 / 7.0);
                        }
                    }
                }
            }
        }

        if (gridOld[offsetZ + dimX + 1] == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        Coord<3> actualDim = dim - Coord<3>(2, 2, 2);
        double updates = 1.0 * maxT * actualDim.prod();
        double gLUPS = 1e-9 * updates / seconds;

        delete gridOld;
        delete gridNew;

        return gLUPS;
    }

    std::string unit()
    {
        return "GLUPS";
    }
};

class Jacobi3DSSE : public CPUBenchmark
{
public:
    std::string family()
    {
        return "Jacobi3D";
    }

    std::string species()
    {
        return "pepper";
    }

    double performance2(const Coord<3>& dim)
    {
        int dimX = dim.x();
        int dimY = dim.y();
        int dimZ = dim.z();
        int offsetZ = dimX * dimY;
        int maxT = 20;

        double *gridOld = new double[dimX * dimY * dimZ];
        double *gridNew = new double[dimX * dimY * dimZ];

        for (int z = 0; z < dimZ; ++z) {
            for (int y = 0; y < dimY; ++y) {
                for (int x = 0; x < dimY; ++x) {
                    gridOld[z * offsetZ + y * dimY + x] = x + y + z;
                    gridNew[z * offsetZ + y * dimY + x] = x + y + z;
                }
            }
        }

        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            for (int t = 0; t < maxT; ++t) {
                for (int z = 1; z < (dimZ - 1); ++z) {
                    for (int y = 1; y < (dimY - 1); ++y) {
                        updateLine(&gridNew[z * offsetZ + y * dimY + 0],
                                   &gridOld[z * offsetZ + y * dimX - offsetZ],
                                   &gridOld[z * offsetZ + y * dimX - dimX],
                                   &gridOld[z * offsetZ + y * dimX + 0],
                                   &gridOld[z * offsetZ + y * dimX + dimX],
                                   &gridOld[z * offsetZ + y * dimX + offsetZ],
                                   1,
                                   dimX - 1);

                    }
                }
            }
        }

        if (gridOld[offsetZ + dimX + 1] == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        Coord<3> actualDim = dim - Coord<3>(2, 2, 2);
        double updates = 1.0 * maxT * actualDim.prod();
        double gLUPS = 1e-9 * updates / seconds;

        delete gridOld;
        delete gridNew;

        return gLUPS;
    }

    std::string unit()
    {
        return "GLUPS";
    }

private:
    void updateLine(
        double *target,
        double *south,
        double *top,
        double *same,
        double *bottom,
        double *north,
        int startX,
        int endX)
    {
        int x = startX;

        if ((x % 2) == 1) {
            updatePoint(target, south, top, same, bottom, north, x);
            ++x;
        }

        __m128d oneSeventh = _mm_set_pd(1.0/7.0, 1.0/7.0);
        __m128d same1 = _mm_load_pd(same + x + 0);
        __m128d odds0 = _mm_loadu_pd(same + x - 1);

        for (; x < (endX - 7); x += 8) {
            __m128d same2 = _mm_load_pd(same + x + 2);
            __m128d same3 = _mm_load_pd(same + x + 4);
            __m128d same4 = _mm_load_pd(same + x + 6);
            __m128d same5 = _mm_load_pd(same + x + 8);

            // shuffle values obtain left/right neighbors
            __m128d odds1 = _mm_shuffle_pd(same1, same2, (1 << 0) | (0 << 2));
            __m128d odds2 = _mm_shuffle_pd(same2, same3, (1 << 0) | (0 << 2));
            __m128d odds3 = _mm_shuffle_pd(same3, same4, (1 << 0) | (0 << 2));
            __m128d odds4 = _mm_shuffle_pd(same4, same5, (1 << 0) | (0 << 2));

            // load south neighbors
            __m128d buf0 =  _mm_load_pd(south + x + 0);
            __m128d buf1 =  _mm_load_pd(south + x + 2);
            __m128d buf2 =  _mm_load_pd(south + x + 4);
            __m128d buf3 =  _mm_load_pd(south + x + 6);

            // add left neighbors
            same1 = _mm_add_pd(same1, odds0);
            same2 = _mm_add_pd(same2, odds1);
            same3 = _mm_add_pd(same3, odds2);
            same4 = _mm_add_pd(same4, odds3);

            // add right neighbors
            same1 = _mm_add_pd(same1, odds1);
            same2 = _mm_add_pd(same2, odds2);
            same3 = _mm_add_pd(same3, odds3);
            same4 = _mm_add_pd(same4, odds4);

            // load top neighbors
            odds0 = _mm_load_pd(top + x + 0);
            odds1 = _mm_load_pd(top + x + 2);
            odds2 = _mm_load_pd(top + x + 4);
            odds3 = _mm_load_pd(top + x + 6);

            // add south neighbors
            same1 = _mm_add_pd(same1, buf0);
            same2 = _mm_add_pd(same2, buf1);
            same3 = _mm_add_pd(same3, buf2);
            same4 = _mm_add_pd(same4, buf3);

            // load bottom neighbors
            buf0 =  _mm_load_pd(bottom + x + 0);
            buf1 =  _mm_load_pd(bottom + x + 2);
            buf2 =  _mm_load_pd(bottom + x + 4);
            buf3 =  _mm_load_pd(bottom + x + 6);

            // add top neighbors
            same1 = _mm_add_pd(same1, odds0);
            same2 = _mm_add_pd(same2, odds1);
            same3 = _mm_add_pd(same3, odds2);
            same4 = _mm_add_pd(same4, odds3);

            // load north neighbors
            odds0 = _mm_load_pd(north + x + 0);
            odds1 = _mm_load_pd(north + x + 2);
            odds2 = _mm_load_pd(north + x + 4);
            odds3 = _mm_load_pd(north + x + 6);

            // add bottom neighbors
            same1 = _mm_add_pd(same1, buf0);
            same2 = _mm_add_pd(same2, buf1);
            same3 = _mm_add_pd(same3, buf2);
            same4 = _mm_add_pd(same4, buf3);

            // add north neighbors
            same1 = _mm_add_pd(same1, odds0);
            same2 = _mm_add_pd(same2, odds1);
            same3 = _mm_add_pd(same3, odds2);
            same4 = _mm_add_pd(same4, odds3);

            // scale by 1/7
            same1 = _mm_mul_pd(same1, oneSeventh);
            same2 = _mm_mul_pd(same2, oneSeventh);
            same3 = _mm_mul_pd(same3, oneSeventh);
            same4 = _mm_mul_pd(same4, oneSeventh);

            // store results
            _mm_store_pd(target + x + 0, same1);
            _mm_store_pd(target + x + 2, same2);
            _mm_store_pd(target + x + 4, same3);
            _mm_store_pd(target + x + 6, same4);

            odds0 = odds4;
            same1 = same5;
        }

        for (; x < endX; ++x) {
            updatePoint(target, south, top, same, bottom, north, x);
        }
    }

    void updatePoint(
        double *target,
        double *south,
        double *top,
        double *same,
        double *bottom,
        double *north,
        int x)
    {
        target[x] =
            (south[x] +
             top[x] +
             same[x - 1] + same[x + 0] + same[x + 1] +
             bottom[x] +
             north[x]) * (1.0 / 7.0);
    }

};

template<typename CELL>
class NoOpInitializer : public SimpleInitializer<CELL>
{
public:
    typedef typename SimpleInitializer<CELL>::Topology Topology;

    NoOpInitializer(
        const Coord<3>& dimensions,
        const unsigned& steps) :
        SimpleInitializer<CELL>(dimensions, steps)
    {}

    virtual void grid(GridBase<CELL, Topology::DIM> *target)
    {}
};

class JacobiCellClassic
{
public:
    class API :
        public APITraits::HasStencil<Stencils::VonNeumann<3, 1> >,
        public APITraits::HasCubeTopology<3>
    {};

    JacobiCellClassic(double t = 0) :
        temp(t)
    {}

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, int /* nanoStep */)
    {
        temp = (hood[Coord<3>( 0,  0, -1)].temp +
                hood[Coord<3>( 0, -1,  0)].temp +
                hood[Coord<3>(-1,  0,  0)].temp +
                hood[Coord<3>( 0,  0,  0)].temp +
                hood[Coord<3>( 1,  0,  0)].temp +
                hood[Coord<3>( 0,  1,  0)].temp +
                hood[Coord<3>( 0,  0,  1)].temp) * (1.0 / 7.0);
    }

    double temp;
};

class Jacobi3DClassic : public CPUBenchmark
{
public:
    std::string family()
    {
        return "Jacobi3D";
    }

    std::string species()
    {
        return "bronze";
    }

    double performance2(const Coord<3>& dim)
    {
        int maxT = 5;
        SerialSimulator<JacobiCellClassic> sim(
            new NoOpInitializer<JacobiCellClassic>(dim, maxT));

        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            sim.run();
        }

        if (sim.getGrid()->get(Coord<3>(1, 1, 1)).temp == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        double updates = 1.0 * maxT * dim.prod();
        double gLUPS = 1e-9 * updates / seconds;

        return gLUPS;
    }

    std::string unit()
    {
        return "GLUPS";
    }
};

class JacobiCellFixedHood
{
public:
    class API :
        public APITraits::HasFixedCoordsOnlyUpdate,
        public APITraits::HasStencil<Stencils::VonNeumann<3, 1> >,
        public APITraits::HasCubeTopology<3>
    {};

    JacobiCellFixedHood(double t = 0) :
        temp(t)
    {}

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, int /* nanoStep */)
    {
        temp = (hood[FixedCoord< 0,  0, -1>()].temp +
                hood[FixedCoord< 0, -1,  0>()].temp +
                hood[FixedCoord<-1,  0,  0>()].temp +
                hood[FixedCoord< 0,  0,  0>()].temp +
                hood[FixedCoord< 1,  0,  0>()].temp +
                hood[FixedCoord< 0,  1,  0>()].temp +
                hood[FixedCoord< 0,  0,  1>()].temp) * (1.0 / 7.0);
    }

    double temp;
};

class Jacobi3DFixedHood : public CPUBenchmark
{
public:
    std::string family()
    {
        return "Jacobi3D";
    }

    std::string species()
    {
        return "silver";
    }

    double performance2(const Coord<3>& dim)
    {
        int maxT = 20;

        SerialSimulator<JacobiCellFixedHood> sim(
            new NoOpInitializer<JacobiCellFixedHood>(dim, maxT));

        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            sim.run();
        }

        if (sim.getGrid()->get(Coord<3>(1, 1, 1)).temp == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        double updates = 1.0 * maxT * dim.prod();
        double gLUPS = 1e-9 * updates / seconds;

        return gLUPS;
    }

    std::string unit()
    {
        return "GLUPS";
    }
};

class QuadM128
{
public:
    __m128d a;
    __m128d b;
    __m128d c;
    __m128d d;
};

class PentaM128
{
public:
    __m128d a;
    __m128d b;
    __m128d c;
    __m128d d;
    __m128d e;
};

class JacobiCellStreakUpdate
{
public:
    class API :
        public APITraits::HasFixedCoordsOnlyUpdate,
        public APITraits::HasUpdateLineX,
        public APITraits::HasStencil<Stencils::VonNeumann<3, 1> >,
        public APITraits::HasCubeTopology<3>,
        public APITraits::HasSoA
    {};

    JacobiCellStreakUpdate(double t = 0) :
        temp(t)
    {}

    template<typename HOOD_OLD, typename HOOD_NEW>
    static void updateSingle(HOOD_OLD& hoodOld, HOOD_NEW& hoodNew)
    {
        hoodNew.temp() =
            (hoodOld[FixedCoord<0,  0, -1>()].temp() +
             hoodOld[FixedCoord<0, -1,  0>()].temp() +
             hoodOld[FixedCoord<1,  0,  0>()].temp() +
             hoodOld[FixedCoord<0,  0,  0>()].temp() +
             hoodOld[FixedCoord<1,  0,  0>()].temp() +
             hoodOld[FixedCoord<0,  1,  0>()].temp() +
             hoodOld[FixedCoord<0,  0,  1>()].temp()) * (1.0 / 7.0);
    }

    template<typename HOOD_OLD, typename HOOD_NEW>
    static void updateLineX(HOOD_OLD& hoodOld, int indexEnd,
                            HOOD_NEW& hoodNew, int /* nanoStep */)
    {
        if (hoodOld.index() % 2 == 1) {
            updateSingle(hoodOld, hoodNew);
            ++hoodOld.index();
            ++hoodNew.index;
        }

        __m128d oneSeventh = _mm_set1_pd(1.0 / 7.0);
        PentaM128 same;
        same.a = _mm_load_pd( &hoodOld[FixedCoord< 0, 0, 0>()].temp());
        __m128d odds0 = _mm_loadu_pd(&hoodOld[FixedCoord<-1, 0, 0>()].temp());

        for (; hoodOld.index() < (indexEnd - 8 + 1); hoodOld.index() += 8, hoodNew.index += 8) {

            load(&same, hoodOld, FixedCoord<0, 0, 0>());

            // shuffle values obtain left/right neighbors
            __m128d odds1 = _mm_shuffle_pd(same.a, same.b, (1 << 0) | (0 << 2));
            __m128d odds2 = _mm_shuffle_pd(same.b, same.c, (1 << 0) | (0 << 2));
            __m128d odds3 = _mm_shuffle_pd(same.c, same.d, (1 << 0) | (0 << 2));
            __m128d odds4 = _mm_shuffle_pd(same.d, same.e, (1 << 0) | (0 << 2));

            // load south neighbors
            QuadM128 buf;
            load(&buf, hoodOld, FixedCoord<0, 0, -1>());

            // add left neighbors
            same.a = _mm_add_pd(same.a, odds0);
            same.b = _mm_add_pd(same.b, odds1);
            same.c = _mm_add_pd(same.c, odds2);
            same.d = _mm_add_pd(same.d, odds3);

            // add right neighbors
            same.a = _mm_add_pd(same.a, odds1);
            same.b = _mm_add_pd(same.b, odds2);
            same.c = _mm_add_pd(same.c, odds3);
            same.d = _mm_add_pd(same.d, odds4);

            // load top neighbors
            odds0 = load<0>(hoodOld, FixedCoord< 0, -1, 0>());
            odds1 = load<2>(hoodOld, FixedCoord< 0, -1, 0>());
            odds2 = load<4>(hoodOld, FixedCoord< 0, -1, 0>());
            odds3 = load<6>(hoodOld, FixedCoord< 0, -1, 0>());

            // add south neighbors
            same.a = _mm_add_pd(same.a, buf.a);
            same.b = _mm_add_pd(same.b, buf.b);
            same.c = _mm_add_pd(same.c, buf.c);
            same.d = _mm_add_pd(same.d, buf.d);

            // load bottom neighbors
            load(&buf, hoodOld, FixedCoord<0, 1, 0>());

            // add top neighbors
            same.a = _mm_add_pd(same.a, odds0);
            same.b = _mm_add_pd(same.b, odds1);
            same.c = _mm_add_pd(same.c, odds2);
            same.d = _mm_add_pd(same.d, odds3);

            // load north neighbors
            odds0 = load<0>(hoodOld, FixedCoord< 0, 0, 1>());
            odds1 = load<2>(hoodOld, FixedCoord< 0, 0, 1>());
            odds2 = load<4>(hoodOld, FixedCoord< 0, 0, 1>());
            odds3 = load<6>(hoodOld, FixedCoord< 0, 0, 1>());

            // add bottom neighbors
            same.a = _mm_add_pd(same.a, buf.a);
            same.b = _mm_add_pd(same.b, buf.b);
            same.c = _mm_add_pd(same.c, buf.c);
            same.d = _mm_add_pd(same.d, buf.d);

            // add north neighbors
            same.a = _mm_add_pd(same.a, odds0);
            same.b = _mm_add_pd(same.b, odds1);
            same.c = _mm_add_pd(same.c, odds2);
            same.d = _mm_add_pd(same.d, odds3);

            // scale by 1/7
            same.a = _mm_mul_pd(same.a, oneSeventh);
            same.b = _mm_mul_pd(same.b, oneSeventh);
            same.c = _mm_mul_pd(same.c, oneSeventh);
            same.d = _mm_mul_pd(same.d, oneSeventh);

            // store results
            _mm_store_pd(&hoodNew[LibFlatArray::coord<0, 0, 0>()].temp(), same.a);
            _mm_store_pd(&hoodNew[LibFlatArray::coord<2, 0, 0>()].temp(), same.b);
            _mm_store_pd(&hoodNew[LibFlatArray::coord<4, 0, 0>()].temp(), same.c);
            _mm_store_pd(&hoodNew[LibFlatArray::coord<6, 0, 0>()].temp(), same.d);

            // cycle members
            odds0 = odds4;
            same.a = same.e;
        }

        for (; hoodOld.index() < indexEnd; ++hoodOld.index(), ++hoodNew.index) {
            updateSingle(hoodOld, hoodNew);
        }
    }

    template<typename NEIGHBORHOOD, int X, int Y, int Z>
    static void load(QuadM128 *q, const NEIGHBORHOOD& hood, FixedCoord<X, Y, Z> coord)
    {
        q->a = load<0>(hood, coord);
        q->b = load<2>(hood, coord);
        q->c = load<4>(hood, coord);
        q->d = load<6>(hood, coord);
    }

    template<typename NEIGHBORHOOD, int X, int Y, int Z>
    static void load(PentaM128 *q, const NEIGHBORHOOD& hood, FixedCoord<X, Y, Z> coord)
    {
        q->b = load<2>(hood, coord);
        q->c = load<4>(hood, coord);
        q->d = load<6>(hood, coord);
        q->e = load<8>(hood, coord);
    }

    template<int OFFSET, typename NEIGHBORHOOD, int X, int Y, int Z>
    static __m128d load(const NEIGHBORHOOD& hood, FixedCoord<X, Y, Z> coord)
    {
        return load<OFFSET>(&hood[coord].temp());
    }

    template<int OFFSET>
    static __m128d load(const double *p)
    {
        return _mm_load_pd(p + OFFSET);
    }

    double temp;
};

LIBFLATARRAY_REGISTER_SOA(
    JacobiCellStreakUpdate,
    ((double)(temp))
                          )

class Jacobi3DStreakUpdate : public CPUBenchmark
{
public:
    std::string family()
    {
        return "Jacobi3D";
    }

    std::string species()
    {
        return "gold";
    }

    double performance2(const Coord<3>& dim)
    {
        typedef SoAGrid<
            JacobiCellStreakUpdate,
            APITraits::SelectTopology<JacobiCellStreakUpdate>::Value> GridType;
        CoordBox<3> box(Coord<3>(), dim);
        GridType gridA(box, 1.0);
        GridType gridB(box, 2.0);
        GridType *gridOld = &gridA;
        GridType *gridNew = &gridB;

        int maxT = 20;

        Region<3> region;
        region << box;

        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            for (int t = 0; t < maxT; ++t) {
                typedef UpdateFunctorHelpers::Selector<JacobiCellStreakUpdate>::SoARegionUpdateHelper Updater;

                Coord<3> offset(1, 1, 1);
                Updater updater(region, offset, offset, 0);
                gridNew->callback(gridOld, updater);
                std::swap(gridOld, gridNew);
            }
        }

        if (gridA.get(Coord<3>(1, 1, 1)).temp == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        double updates = 1.0 * maxT * dim.prod();
        double gLUPS = 1e-9 * updates / seconds;

        return gLUPS;
    }

    std::string unit()
    {
        return "GLUPS";
    }
};

class Jacobi3DStreakUpdateFunctor : public CPUBenchmark
{
public:
    std::string family()
    {
        return "Jacobi3D";
    }

    std::string species()
    {
        return "platinum";
    }

    double performance2(const Coord<3>& dim)
    {
        int maxT = 20;
        SerialSimulator<JacobiCellStreakUpdate> sim(
            new NoOpInitializer<JacobiCellStreakUpdate>(dim, maxT));

        double seconds = 0;
        {
            ScopedTimer t(&seconds);

            sim.run();
        }

        if (sim.getGrid()->get(Coord<3>(1, 1, 1)).temp == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        double updates = 1.0 * maxT * dim.prod();
        double gLUPS = 1e-9 * updates / seconds;

        return gLUPS;
    }

    std::string unit()
    {
        return "GLUPS";
    }
};

class LBMCell
{
public:
    class API :
        public APITraits::HasStencil<Stencils::Moore<3, 1> >,
        public APITraits::HasCubeTopology<3>
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

    inline explicit LBMCell(double v = 1.0, State s = LIQUID) :
        state(s)
    {
        comp[C] = v;
        for (int i = 1; i < 19; ++i) {
            comp[i] = 0.0;
        }
        density = 1.0;
    }

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, const unsigned /* nanoStep */)
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

#undef GET_COMP
#undef SQR
};

class ShortVec1xSSE
{
public:
    static const int ARITY = 2;

    inline ShortVec1xSSE() :
        val(_mm_set1_pd(0))
    {}

    inline ShortVec1xSSE(const double *addr) :
        val(_mm_loadu_pd(addr))
    {}

    inline ShortVec1xSSE(const double val) :
        val(_mm_set1_pd(val))
    {}

    inline ShortVec1xSSE(__m128d val) :
        val(val)
    {}

    inline ShortVec1xSSE operator+(const ShortVec1xSSE a) const
    {
        return ShortVec1xSSE(_mm_add_pd(val, a.val));
    }

    inline ShortVec1xSSE operator-(const ShortVec1xSSE a) const
    {
        return ShortVec1xSSE(_mm_sub_pd(val, a.val));
    }

    inline ShortVec1xSSE operator*(const ShortVec1xSSE a) const
    {
        return ShortVec1xSSE(_mm_mul_pd(val, a.val));
    }

    inline void store(double *a) const
    {
        _mm_store_pd(a + 0, val);
    }

    __m128d val;
};

class ShortVec2xSSE
{
public:
    static const int ARITY = 4;

    inline ShortVec2xSSE() :
        val1(_mm_set1_pd(0)),
        val2(_mm_set1_pd(0))
    {}

    inline ShortVec2xSSE(const double *addr) :
        val1(_mm_loadu_pd(addr + 0)),
        val2(_mm_loadu_pd(addr + 2))
    {}

    inline ShortVec2xSSE(const double val) :
        val1(_mm_set1_pd(val)),
        val2(_mm_set1_pd(val))
    {}

    inline ShortVec2xSSE(__m128d val1, __m128d val2) :
        val1(val1),
        val2(val2)
    {}

    inline ShortVec2xSSE operator+(const ShortVec2xSSE a) const
    {
        return ShortVec2xSSE(
            _mm_add_pd(val1, a.val1),
            _mm_add_pd(val2, a.val2));
    }

    inline ShortVec2xSSE operator-(const ShortVec2xSSE a) const
    {
        return ShortVec2xSSE(
            _mm_sub_pd(val1, a.val1),
            _mm_sub_pd(val2, a.val2));

    }

    inline ShortVec2xSSE operator*(const ShortVec2xSSE a) const
    {
        return ShortVec2xSSE(
            _mm_mul_pd(val1, a.val1),
            _mm_mul_pd(val2, a.val2));
    }

    inline void store(double *a) const
    {
        _mm_store_pd(a + 0, val1);
        _mm_store_pd(a + 2, val2);
    }

    __m128d val1;
    __m128d val2;
};

class ShortVec4xSSE
{
public:
    static const int ARITY = 8;

    inline ShortVec4xSSE() :
        val1(_mm_set1_pd(0)),
        val2(_mm_set1_pd(0)),
        val3(_mm_set1_pd(0)),
        val4(_mm_set1_pd(0))
    {}

    inline ShortVec4xSSE(const double *addr) :
        val1(_mm_loadu_pd(addr + 0)),
        val2(_mm_loadu_pd(addr + 2)),
        val3(_mm_loadu_pd(addr + 4)),
        val4(_mm_loadu_pd(addr + 6))
    {}

    inline ShortVec4xSSE(const double val) :
        val1(_mm_set1_pd(val)),
        val2(_mm_set1_pd(val)),
        val3(_mm_set1_pd(val)),
        val4(_mm_set1_pd(val))
    {}

    inline ShortVec4xSSE(__m128d val1, __m128d val2, __m128d val3, __m128d val4) :
        val1(val1),
        val2(val2),
        val3(val3),
        val4(val4)
    {}

    inline ShortVec4xSSE operator+(const ShortVec4xSSE a) const
    {
        return ShortVec4xSSE(
            _mm_add_pd(val1, a.val1),
            _mm_add_pd(val2, a.val2),
            _mm_add_pd(val3, a.val3),
            _mm_add_pd(val4, a.val4));
    }

    inline ShortVec4xSSE operator-(const ShortVec4xSSE a) const
    {
        return ShortVec4xSSE(
            _mm_sub_pd(val1, a.val1),
            _mm_sub_pd(val2, a.val2),
            _mm_sub_pd(val3, a.val3),
            _mm_sub_pd(val4, a.val4));

    }

    inline ShortVec4xSSE operator*(const ShortVec4xSSE a) const
    {
        return ShortVec4xSSE(
            _mm_mul_pd(val1, a.val1),
            _mm_mul_pd(val2, a.val2),
            _mm_mul_pd(val3, a.val3),
            _mm_mul_pd(val4, a.val4));
    }

    inline void store(double *a) const
    {
        _mm_storeu_pd(a + 0, val1);
        _mm_storeu_pd(a + 2, val2);
        _mm_storeu_pd(a + 4, val3);
        _mm_storeu_pd(a + 6, val4);
    }

private:
    __m128d val1;
    __m128d val2;
    __m128d val3;
    __m128d val4;
};

#ifdef __AVX__

class ShortVec1xAVX
{
public:
    static const int ARITY = 4;

    inline ShortVec1xAVX() :
        val1(_mm256_set_pd(0, 0, 0, 0))
    {}

    inline ShortVec1xAVX(const double *addr) :
        val1(_mm256_loadu_pd(addr +  0))
    {}

    inline ShortVec1xAVX(const double val) :
        val1(_mm256_set_pd(val, val, val, val))
    {}

    inline ShortVec1xAVX(__m256d val1) :
        val1(val1)
    {}

    inline ShortVec1xAVX operator+(const ShortVec1xAVX a) const
    {
        return ShortVec1xAVX(
            _mm256_add_pd(val1, a.val1));
    }

    inline ShortVec1xAVX operator-(const ShortVec1xAVX a) const
    {
        return ShortVec1xAVX(
            _mm256_sub_pd(val1, a.val1));

    }

    inline ShortVec1xAVX operator*(const ShortVec1xAVX a) const
    {
        return ShortVec1xAVX(
            _mm256_mul_pd(val1, a.val1));
    }

    inline void store(double *a) const
    {
        _mm256_storeu_pd(a +  0, val1);
    }

private:
    __m256d val1;
};

class ShortVec2xAVX
{
public:
    static const int ARITY = 8;

    inline ShortVec2xAVX() :
        val1(_mm256_set_pd(0, 0, 0, 0)),
        val2(_mm256_set_pd(0, 0, 0, 0))
    {}

    inline ShortVec2xAVX(const double *addr) :
        val1(_mm256_loadu_pd(addr +  0)),
        val2(_mm256_loadu_pd(addr +  4))
    {}

    inline ShortVec2xAVX(const double val) :
        val1(_mm256_set_pd(val, val, val, val)),
        val2(_mm256_set_pd(val, val, val, val))
    {}

    inline ShortVec2xAVX(__m256d val1, __m256d val2) :
        val1(val1),
        val2(val2)
    {}

    inline ShortVec2xAVX operator+(const ShortVec2xAVX a) const
    {
        return ShortVec2xAVX(
            _mm256_add_pd(val1, a.val1),
            _mm256_add_pd(val2, a.val2));
    }

    inline ShortVec2xAVX operator-(const ShortVec2xAVX a) const
    {
        return ShortVec2xAVX(
            _mm256_sub_pd(val1, a.val1),
            _mm256_sub_pd(val2, a.val2));

    }

    inline ShortVec2xAVX operator*(const ShortVec2xAVX a) const
    {
        return ShortVec2xAVX(
            _mm256_mul_pd(val1, a.val1),
            _mm256_mul_pd(val2, a.val2));
    }

    inline void store(double *a) const
    {
        _mm256_storeu_pd(a +  0, val1);
        _mm256_storeu_pd(a +  4, val2);
    }

private:
    __m256d val1;
    __m256d val2;
};

class ShortVec4xAVX
{
public:
    static const int ARITY = 16;

    inline ShortVec4xAVX() :
        val1(_mm256_set_pd(0, 0, 0, 0)),
        val2(_mm256_set_pd(0, 0, 0, 0)),
        val3(_mm256_set_pd(0, 0, 0, 0)),
        val4(_mm256_set_pd(0, 0, 0, 0))
    {}

    inline ShortVec4xAVX(const double *addr) :
        val1(_mm256_loadu_pd(addr +  0)),
        val2(_mm256_loadu_pd(addr +  4)),
        val3(_mm256_loadu_pd(addr +  8)),
        val4(_mm256_loadu_pd(addr + 12))
    {}

    inline ShortVec4xAVX(const double val) :
        val1(_mm256_set_pd(val, val, val, val)),
        val2(_mm256_set_pd(val, val, val, val)),
        val3(_mm256_set_pd(val, val, val, val)),
        val4(_mm256_set_pd(val, val, val, val))
    {}

    inline ShortVec4xAVX(__m256d val1, __m256d val2, __m256d val3, __m256d val4) :
        val1(val1),
        val2(val2),
        val3(val3),
        val4(val4)
    {}

    inline ShortVec4xAVX operator+(const ShortVec4xAVX a) const
    {
        return ShortVec4xAVX(
            _mm256_add_pd(val1, a.val1),
            _mm256_add_pd(val2, a.val2),
            _mm256_add_pd(val3, a.val3),
            _mm256_add_pd(val4, a.val4));
    }

    inline ShortVec4xAVX operator-(const ShortVec4xAVX a) const
    {
        return ShortVec4xAVX(
            _mm256_sub_pd(val1, a.val1),
            _mm256_sub_pd(val2, a.val2),
            _mm256_sub_pd(val3, a.val3),
            _mm256_sub_pd(val4, a.val4));

    }

    inline ShortVec4xAVX operator*(const ShortVec4xAVX a) const
    {
        return ShortVec4xAVX(
            _mm256_mul_pd(val1, a.val1),
            _mm256_mul_pd(val2, a.val2),
            _mm256_mul_pd(val3, a.val3),
            _mm256_mul_pd(val4, a.val4));
    }

    inline void store(double *a) const
    {
        _mm256_storeu_pd(a +  0, val1);
        _mm256_storeu_pd(a +  4, val2);
        _mm256_storeu_pd(a +  8, val3);
        _mm256_storeu_pd(a + 12, val4);
    }

private:
    __m256d val1;
    __m256d val2;
    __m256d val3;
    __m256d val4;
};

#endif

template<typename VEC>
void store(double *a, VEC v)
{
    v.store(a);
}

class LBMSoACell
{
public:
    // typedef ShortVec1xSSE Double;
    // typedef ShortVec2xSSE Double;
    typedef ShortVec4xSSE Double;
    // typedef ShortVec1xAVX Double;
    // typedef ShortVec2xAVX Double;
    // typedef ShortVec4xAVX Double;

    class API : public APITraits::HasFixedCoordsOnlyUpdate,
                public APITraits::HasSoA,
                public APITraits::HasUpdateLineX,
                public APITraits::HasStencil<Stencils::Moore<3, 1> >,
                public APITraits::HasCubeTopology<3>
    {};

    enum State {LIQUID, WEST_NOSLIP, EAST_NOSLIP, TOP, BOTTOM, NORTH_ACC, SOUTH_NOSLIP};

    inline explicit LBMSoACell(const double& v=1.0, const State& s=LIQUID) :
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
    {
    }

    template<typename ACCESSOR1, typename ACCESSOR2>
    static void updateLineX(
        ACCESSOR1& hoodOld, int indexEnd,
        ACCESSOR2& hoodNew, int nanoStep)
    {
        updateLineXFluid(hoodOld, indexEnd, hoodNew);
    }



//     template<typename COORD_MAP>
//     void update(const COORD_MAP& neighborhood, const unsigned& nanoStep)
//     {
//         *this = neighborhood[FixedCoord<0, 0>()];

//         switch (state) {
//         case LIQUID:
//             updateFluid(neighborhood);
//             break;
//         case WEST_NOSLIP:
//             updateWestNoSlip(neighborhood);
//             break;
//         case EAST_NOSLIP:
//             updateEastNoSlip(neighborhood);
//             break;
//         case TOP :
//             updateTop(neighborhood);
//             break;
//         case BOTTOM:
//             updateBottom(neighborhood);
//             break;
//         case NORTH_ACC:
//             updateNorthAcc(neighborhood);
//             break;
//         case SOUTH_NOSLIP:
//             updateSouthNoSlip(neighborhood);
//             break;
//         }
//     }

// private:
    template<typename ACCESSOR1, typename ACCESSOR2>
    static void updateLineXFluid(
        ACCESSOR1& hoodOld, int indexEnd,
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

        for (; hoodOld.index() < indexEnd; hoodNew.index += Double::ARITY, hoodOld.index() += Double::ARITY) {
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

            store(&hoodNew.density(), rho);
            store(&hoodNew.velocityX(), velX);
            store(&hoodNew.velocityY(), velY);
            store(&hoodNew.velocityZ(), velZ);

            const Double dir_indep_trm = one_third*rho - one_half *( velX*velX + velY*velY + velZ*velZ );

            store(&hoodNew.C(), omega_trm * GET_COMP(x,y,z,C) + omega_w0*( dir_indep_trm ));

            store(&hoodNew.NW(), omega_trm * GET_COMP(x+1,y-1,z,NW) + omega_w2*( dir_indep_trm - ( velX-velY ) + one_point_five * SQR( velX-velY ) ));
            store(&hoodNew.SE(), omega_trm * GET_COMP(x-1,y+1,z,SE) + omega_w2*( dir_indep_trm + ( velX-velY ) + one_point_five * SQR( velX-velY ) ));
            store(&hoodNew.NE(), omega_trm * GET_COMP(x-1,y-1,z,NE) + omega_w2*( dir_indep_trm + ( velX+velY ) + one_point_five * SQR( velX+velY ) ));
            store(&hoodNew.SW(), omega_trm * GET_COMP(x+1,y+1,z,SW) + omega_w2*( dir_indep_trm - ( velX+velY ) + one_point_five * SQR( velX+velY ) ));

            store(&hoodNew.TW(), omega_trm * GET_COMP(x+1,y,z-1,TW) + omega_w2*( dir_indep_trm - ( velX-velZ ) + one_point_five * SQR( velX-velZ ) ));
            store(&hoodNew.BE(), omega_trm * GET_COMP(x-1,y,z+1,BE) + omega_w2*( dir_indep_trm + ( velX-velZ ) + one_point_five * SQR( velX-velZ ) ));
            store(&hoodNew.TE(), omega_trm * GET_COMP(x-1,y,z-1,TE) + omega_w2*( dir_indep_trm + ( velX+velZ ) + one_point_five * SQR( velX+velZ ) ));
            store(&hoodNew.BW(), omega_trm * GET_COMP(x+1,y,z+1,BW) + omega_w2*( dir_indep_trm - ( velX+velZ ) + one_point_five * SQR( velX+velZ ) ));

            store(&hoodNew.TS(), omega_trm * GET_COMP(x,y+1,z-1,TS) + omega_w2*( dir_indep_trm - ( velY-velZ ) + one_point_five * SQR( velY-velZ ) ));
            store(&hoodNew.BN(), omega_trm * GET_COMP(x,y-1,z+1,BN) + omega_w2*( dir_indep_trm + ( velY-velZ ) + one_point_five * SQR( velY-velZ ) ));
            store(&hoodNew.TN(), omega_trm * GET_COMP(x,y-1,z-1,TN) + omega_w2*( dir_indep_trm + ( velY+velZ ) + one_point_five * SQR( velY+velZ ) ));
            store(&hoodNew.BS(), omega_trm * GET_COMP(x,y+1,z+1,BS) + omega_w2*( dir_indep_trm - ( velY+velZ ) + one_point_five * SQR( velY+velZ ) ));

            store(&hoodNew.N(), omega_trm * GET_COMP(x,y-1,z,N) + omega_w1*( dir_indep_trm + velY + one_point_five * SQR(velY)));
            store(&hoodNew.S(), omega_trm * GET_COMP(x,y+1,z,S) + omega_w1*( dir_indep_trm - velY + one_point_five * SQR(velY)));
            store(&hoodNew.E(), omega_trm * GET_COMP(x-1,y,z,E) + omega_w1*( dir_indep_trm + velX + one_point_five * SQR(velX)));
            store(&hoodNew.W(), omega_trm * GET_COMP(x+1,y,z,W) + omega_w1*( dir_indep_trm - velX + one_point_five * SQR(velX)));
            store(&hoodNew.T(), omega_trm * GET_COMP(x,y,z-1,T) + omega_w1*( dir_indep_trm + velZ + one_point_five * SQR(velZ)));
            store(&hoodNew.B(), omega_trm * GET_COMP(x,y,z+1,B) + omega_w1*( dir_indep_trm - velZ + one_point_five * SQR(velZ)));
        }
    }

//     template<typename ACCESSOR1, typename ACCESSOR2>
//     static void updateLineXFluid(
//         ACCESSOR1 hoodOld, int hoodOld.index(), int indexEnd, ACCESSOR2 hoodNew, int *indexNew)
//     {
// #define GET_COMP(X, Y, Z, COMP) hoodOld[FixedCoord<X, Y, Z>()].COMP()
// #define SQR(X) ((X)*(X))
//         const double omega = 1.0/1.7;
//         const double omega_trm = 1.0 - omega;
//         const double omega_w0 = 3.0 * 1.0 / 3.0 * omega;
//         const double omega_w1 = 3.0*1.0/18.0*omega;
//         const double omega_w2 = 3.0*1.0/36.0*omega;
//         const double one_third = 1.0 / 3.0;
//         const int x = 0;
//         const int y = 0;
//         const int z = 0;
//         double velX, velY, velZ;

//         for (; hoodOld.index() < indexEnd; ++hoodOld.index()) {
//             velX  =
//                 GET_COMP(x-1,y,z,E) + GET_COMP(x-1,y-1,z,NE) +
//                 GET_COMP(x-1,y+1,z,SE) + GET_COMP(x-1,y,z-1,TE) +
//                 GET_COMP(x-1,y,z+1,BE);
//             velY  = GET_COMP(x,y-1,z,N) + GET_COMP(x+1,y-1,z,NW) +
//                 GET_COMP(x,y-1,z-1,TN) + GET_COMP(x,y-1,z+1,BN);
//             velZ  = GET_COMP(x,y,z-1,T) + GET_COMP(x,y+1,z-1,TS) +
//                 GET_COMP(x+1,y,z-1,TW);

//             const double rho =
//                 GET_COMP(x,y,z,C) + GET_COMP(x,y+1,z,S) +
//                 GET_COMP(x+1,y,z,W) + GET_COMP(x,y,z+1,B) +
//                 GET_COMP(x+1,y+1,z,SW) + GET_COMP(x,y+1,z+1,BS) +
//                 GET_COMP(x+1,y,z+1,BW) + velX + velY + velZ;
//             velX  = velX
//                 - GET_COMP(x+1,y,z,W)    - GET_COMP(x+1,y-1,z,NW)
//                 - GET_COMP(x+1,y+1,z,SW) - GET_COMP(x+1,y,z-1,TW)
//                 - GET_COMP(x+1,y,z+1,BW);
//             velY  = velY
//                 + GET_COMP(x-1,y-1,z,NE) - GET_COMP(x,y+1,z,S)
//                 - GET_COMP(x+1,y+1,z,SW) - GET_COMP(x-1,y+1,z,SE)
//                 - GET_COMP(x,y+1,z-1,TS) - GET_COMP(x,y+1,z+1,BS);
//             velZ  = velZ+GET_COMP(x,y-1,z-1,TN) + GET_COMP(x-1,y,z-1,TE) - GET_COMP(x,y,z+1,B) - GET_COMP(x,y-1,z+1,BN) - GET_COMP(x,y+1,z+1,BS) - GET_COMP(x+1,y,z+1,BW) - GET_COMP(x-1,y,z+1,BE);

//             hoodNew.density() = rho;
//             hoodNew.velocityX() = velX;
//             hoodNew.velocityY() = velY;
//             hoodNew.velocityZ() = velZ;

//             const double dir_indep_trm = one_third*rho - 0.5*( velX*velX + velY*velY + velZ*velZ );

//             hoodNew.C()=omega_trm * GET_COMP(x,y,z,C) + omega_w0*( dir_indep_trm );

//             hoodNew.NW()=omega_trm * GET_COMP(x+1,y-1,z,NW) +
//                 omega_w2*( dir_indep_trm - ( velX-velY ) + 1.5*SQR( velX-velY ) );
//             hoodNew.SE()=omega_trm * GET_COMP(x-1,y+1,z,SE) +
//                 omega_w2*( dir_indep_trm + ( velX-velY ) + 1.5*SQR( velX-velY ) );
//             hoodNew.NE()=omega_trm * GET_COMP(x-1,y-1,z,NE) +
//                 omega_w2*( dir_indep_trm + ( velX+velY ) + 1.5*SQR( velX+velY ) );
//             hoodNew.SW()=omega_trm * GET_COMP(x+1,y+1,z,SW) +
//                 omega_w2*( dir_indep_trm - ( velX+velY ) + 1.5*SQR( velX+velY ) );

//             hoodNew.TW()=omega_trm * GET_COMP(x+1,y,z-1,TW) + omega_w2*( dir_indep_trm - ( velX-velZ ) + 1.5*SQR( velX-velZ ) );
//             hoodNew.BE()=omega_trm * GET_COMP(x-1,y,z+1,BE) + omega_w2*( dir_indep_trm + ( velX-velZ ) + 1.5*SQR( velX-velZ ) );
//             hoodNew.TE()=omega_trm * GET_COMP(x-1,y,z-1,TE) + omega_w2*( dir_indep_trm + ( velX+velZ ) + 1.5*SQR( velX+velZ ) );
//             hoodNew.BW()=omega_trm * GET_COMP(x+1,y,z+1,BW) + omega_w2*( dir_indep_trm - ( velX+velZ ) + 1.5*SQR( velX+velZ ) );

//             hoodNew.TS()=omega_trm * GET_COMP(x,y+1,z-1,TS) + omega_w2*( dir_indep_trm - ( velY-velZ ) + 1.5*SQR( velY-velZ ) );
//             hoodNew.BN()=omega_trm * GET_COMP(x,y-1,z+1,BN) + omega_w2*( dir_indep_trm + ( velY-velZ ) + 1.5*SQR( velY-velZ ) );
//             hoodNew.TN()=omega_trm * GET_COMP(x,y-1,z-1,TN) + omega_w2*( dir_indep_trm + ( velY+velZ ) + 1.5*SQR( velY+velZ ) );
//             hoodNew.BS()=omega_trm * GET_COMP(x,y+1,z+1,BS) + omega_w2*( dir_indep_trm - ( velY+velZ ) + 1.5*SQR( velY+velZ ) );

//             hoodNew.N()=omega_trm * GET_COMP(x,y-1,z,N) + omega_w1*( dir_indep_trm + velY + 1.5*SQR(velY));
//             hoodNew.S()=omega_trm * GET_COMP(x,y+1,z,S) + omega_w1*( dir_indep_trm - velY + 1.5*SQR(velY));
//             hoodNew.E()=omega_trm * GET_COMP(x-1,y,z,E) + omega_w1*( dir_indep_trm + velX + 1.5*SQR(velX));
//             hoodNew.W()=omega_trm * GET_COMP(x+1,y,z,W) + omega_w1*( dir_indep_trm - velX + 1.5*SQR(velX));
//             hoodNew.T()=omega_trm * GET_COMP(x,y,z-1,T) + omega_w1*( dir_indep_trm + velZ + 1.5*SQR(velZ));
//             hoodNew.B()=omega_trm * GET_COMP(x,y,z+1,B) + omega_w1*( dir_indep_trm - velZ + 1.5*SQR(velZ));

//             ++*indexNew;
//         }
//     }

//     template<typename COORD_MAP>
//     void updateWestNoSlip(const COORD_MAP& neighborhood)
//     {
//         comp[E ]=GET_COMP(1, 0,  0, W);
//         comp[NE]=GET_COMP(1, 1,  0, SW);
//         comp[SE]=GET_COMP(1,-1,  0, NW);
//         comp[TE]=GET_COMP(1, 0,  1, BW);
//         comp[BE]=GET_COMP(1, 0, -1, TW);
//     }

//     template<typename COORD_MAP>
//     void updateEastNoSlip(const COORD_MAP& neighborhood)
//     {
//         comp[W ]=GET_COMP(-1, 0, 0, E);
//         comp[NW]=GET_COMP(-1, 0, 1, SE);
//         comp[SW]=GET_COMP(-1,-1, 0, NE);
//         comp[TW]=GET_COMP(-1, 0, 1, BE);
//         comp[BW]=GET_COMP(-1, 0,-1, TE);
//     }

//     template<typename COORD_MAP>
//     void updateTop(const COORD_MAP& neighborhood)
//     {
//         comp[B] =GET_COMP(0,0,-1,T);
//         comp[BE]=GET_COMP(1,0,-1,TW);
//         comp[BW]=GET_COMP(-1,0,-1,TE);
//         comp[BN]=GET_COMP(0,1,-1,TS);
//         comp[BS]=GET_COMP(0,-1,-1,TN);
//     }

//     template<typename COORD_MAP>
//     void updateBottom(const COORD_MAP& neighborhood)
//     {
//         comp[T] =GET_COMP(0,0,1,B);
//         comp[TE]=GET_COMP(1,0,1,BW);
//         comp[TW]=GET_COMP(-1,0,1,BE);
//         comp[TN]=GET_COMP(0,1,1,BS);
//         comp[TS]=GET_COMP(0,-1,1,BN);
//     }

//     template<typename COORD_MAP>
//     void updateNorthAcc(const COORD_MAP& neighborhood)
//     {
//         const double w_1 = 0.01;
//         comp[S] =GET_COMP(0,-1,0,N);
//         comp[SE]=GET_COMP(1,-1,0,NW)+6*w_1*0.1;
//         comp[SW]=GET_COMP(-1,-1,0,NE)-6*w_1*0.1;
//         comp[TS]=GET_COMP(0,-1,1,BN);
//         comp[BS]=GET_COMP(0,-1,-1,TN);
//     }

//     template<typename COORD_MAP>
//     void updateSouthNoSlip(const COORD_MAP& neighborhood)
//     {
//         comp[N] =GET_COMP(0,1,0,S);
//         comp[NE]=GET_COMP(1,1,0,SW);
//         comp[NW]=GET_COMP(-1,1,0,SE);
//         comp[TN]=GET_COMP(0,1,1,BS);
//         comp[BN]=GET_COMP(0,1,-1,TS);
//     }

//     double comp[19];
//     double density;
//     double velocityX;
//     double velocityY;
//     double velocityZ;
//     State state;


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

class LBMClassic : public CPUBenchmark
{
public:
    std::string family()
    {
        return "LBM";
    }

    std::string species()
    {
        return "bronze";
    }

    double performance2(const Coord<3>& dim)
    {
        int maxT = 20;
        SerialSimulator<LBMCell> sim(
            new NoOpInitializer<LBMCell>(dim, maxT));

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

    std::string unit()
    {
        return "GLUPS";
    }
};

class LBMSoA : public CPUBenchmark
{
public:
    std::string family()
    {
        return "LBM";
    }

    std::string species()
    {
        return "gold";
    }

    double performance2(const Coord<3>& dim)
    {
        int maxT = 10;
        SerialSimulator<LBMSoACell> sim(
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

    std::string unit()
    {
        return "GLUPS";
    }
};

template<class PARTITION>
class PartitionBenchmark : public CPUBenchmark
{
public:
    PartitionBenchmark(const std::string& name) :
        name(name)
    {}

    std::string species()
    {
        return "gold";
    }

    std::string family()
    {
        return name;
    }

    double performance2(const Coord<3>& dim)
    {
        double duration = 0;
        Coord<2> accu;
        Coord<2> realDim(dim.x(), dim.y());

        {
            ScopedTimer t(&duration);

            PARTITION h(Coord<2>(100, 200), realDim);
            typename PARTITION::Iterator end = h.end();
            for (typename PARTITION::Iterator i = h.begin(); i != end; ++i) {
                accu += *i;
            }
        }

        if (accu == Coord<2>()) {
            throw std::runtime_error("oops, partition iteration went bad!");
        }

        return duration;
    }

    std::string unit()
    {
        return "s";
    }

private:
    std::string name;
};

#ifdef LIBGEODECOMP_WITH_CUDA
void cudaTests(std::string revision, bool quick, int cudaDevice);
#endif

int main(int argc, char **argv)
{
    if ((argc < 3) || (argc > 4)) {
        std::cerr << "usage: " << argv[0] << " [-q,--quick] REVISION CUDA_DEVICE\n";
        return 1;
    }

    bool quick = false;
    int argumentIndex = 1;
    if (argc == 4) {
        if ((std::string(argv[1]) == "-q") ||
            (std::string(argv[1]) == "--quick")) {
            quick = true;
        }
        argumentIndex = 2;
    }
    std::string revision = argv[argumentIndex + 0];

    std::stringstream s;
    s << argv[argumentIndex + 1];
    int cudaDevice;
    s >> cudaDevice;

    LibFlatArray::evaluate eval(revision);
    eval.print_header();

    eval(RegionCount(), toVector(Coord<3>( 128,  128,  128)));
    eval(RegionCount(), toVector(Coord<3>( 512,  512,  512)));
    eval(RegionCount(), toVector(Coord<3>(2048, 2048, 2048)));

    eval(RegionInsert(), toVector(Coord<3>( 128,  128,  128)));
    eval(RegionInsert(), toVector(Coord<3>( 512,  512,  512)));
    eval(RegionInsert(), toVector(Coord<3>(2048, 2048, 2048)));

    eval(RegionIntersect(), toVector(Coord<3>( 128,  128,  128)));
    eval(RegionIntersect(), toVector(Coord<3>( 512,  512,  512)));
    eval(RegionIntersect(), toVector(Coord<3>(2048, 2048, 2048)));

    eval(CoordEnumerationVanilla(), toVector(Coord<3>( 128,  128,  128)));
    eval(CoordEnumerationVanilla(), toVector(Coord<3>( 512,  512,  512)));
    eval(CoordEnumerationVanilla(), toVector(Coord<3>(2048, 2048, 2048)));

    eval(CoordEnumerationBronze(), toVector(Coord<3>( 128,  128,  128)));
    eval(CoordEnumerationBronze(), toVector(Coord<3>( 512,  512,  512)));
    eval(CoordEnumerationBronze(), toVector(Coord<3>(2048, 2048, 2048)));

    eval(CoordEnumerationGold(), toVector(Coord<3>( 128,  128,  128)));
    eval(CoordEnumerationGold(), toVector(Coord<3>( 512,  512,  512)));
    eval(CoordEnumerationGold(), toVector(Coord<3>(2048, 2048, 2048)));

    eval(FloatCoordAccumulationGold(), toVector(Coord<3>(2048, 2048, 2048)));

    std::vector<Coord<3> > sizes;
    sizes << Coord<3>(22, 22, 22)
          << Coord<3>(64, 64, 64)
          << Coord<3>(68, 68, 68)
          << Coord<3>(106, 106, 106)
          << Coord<3>(128, 128, 128)
          << Coord<3>(150, 150, 150)
          << Coord<3>(512, 512, 32)
          << Coord<3>(518, 518, 32)
          << Coord<3>(1024, 1024, 32)
          << Coord<3>(1026, 1026, 32);

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(Jacobi3DVanilla(), toVector(sizes[i]));
    }

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(Jacobi3DSSE(), toVector(sizes[i]));
    }

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(Jacobi3DClassic(), toVector(sizes[i]));
    }

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(Jacobi3DFixedHood(), toVector(sizes[i]));
    }

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(Jacobi3DStreakUpdate(), toVector(sizes[i]));
    }

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(Jacobi3DStreakUpdateFunctor(), toVector(sizes[i]));
    }

    sizes.clear();

    sizes << Coord<3>(22, 22, 22)
          << Coord<3>(64, 64, 64)
          << Coord<3>(68, 68, 68)
          << Coord<3>(106, 106, 106)
          << Coord<3>(128, 128, 128)
          << Coord<3>(160, 160, 160);

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(LBMClassic(), toVector(sizes[i]));
    }

    for (std::size_t i = 0; i < sizes.size(); ++i) {
        eval(LBMSoA(), toVector(sizes[i]));
    }

    std::vector<int> dim = toVector(Coord<3>(32 * 1024, 32 * 1024, 1));
    eval(PartitionBenchmark<HIndexingPartition   >("PartitionHIndexing"), dim);
    eval(PartitionBenchmark<StripingPartition<2> >("PartitionStriping"),  dim);
    eval(PartitionBenchmark<HilbertPartition     >("PartitionHilbert"),   dim);
    eval(PartitionBenchmark<ZCurvePartition<2>   >("PartitionZCurve"),    dim);

#ifdef LIBGEODECOMP_WITH_CUDA
    cudaTests(revision, quick, cudaDevice);
#endif

    return 0;
}
