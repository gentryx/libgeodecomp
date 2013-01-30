#include <boost/date_time/posix_time/posix_time.hpp>
#include <emmintrin.h>
#include <iomanip>
#include <iostream>
#include <libgeodecomp/config.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/misc/cellapitraits.h>
#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/misc/linepointerassembly.h>
#include <libgeodecomp/misc/linepointerupdatefunctor.h>
#include <libgeodecomp/misc/region.h>
#include <libgeodecomp/misc/soaaccessor.h>
#include <libgeodecomp/misc/stencils.h>
#include <libgeodecomp/misc/updatefunctor.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <stdio.h>

using namespace LibGeoDecomp;

std::string revision;

class RegionCount
{
public:
    std::string order()
    {
        return "CPU";
    }

    std::string family()
    {
        return "RegionCount";
    }

    std::string species()
    {
        return "gold";
    }

    double performance(const Coord<3>& dim)
    {
        int sum = 0;
        Region<3> r;
        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                r << Streak<3>(Coord<3>(0, y, z), dim.x());
            }
        }

        long long tStart = Chronometer::timeUSec();

        for (int z = 0; z < dim.z(); z += 4) {
            for (int y = 0; y < dim.y(); y += 4) {
                for (int x = 0; x < dim.x(); x += 4) {
                    sum += r.count(Coord<3>(x, y, z));
                }
            }
        }

        long long tEnd = Chronometer::timeUSec();

        if (sum == 31) {
            std::cout << "pure debug statement to prevent the compiler from optimizing away the previous loop";
        }

        return (tEnd - tStart) * 0.000001;
    }

    std::string unit()
    {
        return "s";
    }
};

class RegionInsert
{
public:
    std::string order()
    {
        return "CPU";
    }

    std::string family()
    {
        return "RegionInsert";
    }

    std::string species()
    {
        return "gold";
    }

    double performance(const Coord<3>& dim)
    {
        long long tStart = Chronometer::timeUSec();

        Region<3> r;
        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                r << Streak<3>(Coord<3>(0, y, z), dim.x());
            }
        }

        long long tEnd = Chronometer::timeUSec();

        return (tEnd - tStart) * 0.000001;
    }

    std::string unit()
    {
        return "s";
    }
};

class RegionIntersect
{
public:
    std::string order()
    {
        return "CPU";
    }

    std::string family()
    {
        return "RegionIntersect";
    }

    std::string species()
    {
        return "gold";
    }

    double performance(const Coord<3>& dim)
    {
        long long tStart = Chronometer::timeUSec();

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

        long long tEnd = Chronometer::timeUSec();

        return (tEnd - tStart) * 0.000001;
    }

    std::string unit()
    {
        return "s";
    }
};

class CoordEnumerationVanilla
{
public:
    std::string order()
    {
        return "CPU";
    }

    std::string family()
    {
        return "CoordEnumeration";
    }

    std::string species()
    {
        return "vanilla";
    }

    double performance(const Coord<3>& dim)
    {
        long long tStart = Chronometer::timeUSec();
        
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

        long long tEnd = Chronometer::timeUSec();

        return (tEnd - tStart) * 0.000001;
    }

    std::string unit()
    {
        return "s";
    }
};

class CoordEnumerationBronze
{
public:
    std::string order()
    {
        return "CPU";
    }

    std::string family()
    {
        return "CoordEnumeration";
    }

    std::string species()
    {
        return "bronze";
    }

    double performance(const Coord<3>& dim)
    {
        Region<3> region;
        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                region << Streak<3>(Coord<3>(0, y, z), dim.x());
            }
        }
        long long tStart = Chronometer::timeUSec();
        
        Coord<3> sum;
        for (Region<3>::Iterator i = region.begin(); i != region.end(); ++i) {
            sum += *i;
        }
        // trick the compiler to not optimize away the loop above
        if (sum == Coord<3>(1, 2, 3)) {
            std::cout << "whatever";
        }

        long long tEnd = Chronometer::timeUSec();

        return (tEnd - tStart) * 0.000001;
    }

    std::string unit()
    {
        return "s";
    }
};

class CoordEnumerationGold
{
public:
    std::string order()
    {
        return "CPU";
    }

    std::string family()
    {
        return "CoordEnumeration";
    }

    std::string species()
    {
        return "gold";
    }

    double performance(const Coord<3>& dim)
    {
        Region<3> region;
        for (int z = 0; z < dim.z(); ++z) {
            for (int y = 0; y < dim.y(); ++y) {
                region << Streak<3>(Coord<3>(0, y, z), dim.x());
            }
        }
        long long tStart = Chronometer::timeUSec();
        
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

        long long tEnd = Chronometer::timeUSec();

        return (tEnd - tStart) * 0.000001;
    }

    std::string unit()
    {
        return "s";
    }
};

class Jacobi3DVanilla
{
public:
    std::string order()
    {
        return "CPU";
    }

    std::string family()
    {
        return "Jacobi3D";
    }

    std::string species()
    {
        return "vanilla";
    }

    double performance(const Coord<3>& dim)
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

        long long tBegin= Chronometer::timeUSec();

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

        long long tEnd = Chronometer::timeUSec();

        if (gridOld[offsetZ + dimX + 1] == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        Coord<3> actualDim = dim - Coord<3>(2, 2, 2);
        double updates = 1.0 * maxT * actualDim.prod();
        double seconds = (tEnd - tBegin) * 10e-6;
        double gLUPS = 10e-9 * updates / seconds;

        delete gridOld;
        delete gridNew;

        return gLUPS;
    }

    std::string unit()
    {
        return "GLUPS";
    }
};

class Jacobi3DSSE
{
public:
    std::string order()
    {
        return "CPU";
    }

    std::string family()
    {
        return "Jacobi3D";
    }

    std::string species()
    {
        return "pepper";
    }

    double performance(const Coord<3>& dim)
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

        long long tBegin= Chronometer::timeUSec();

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

        long long tEnd = Chronometer::timeUSec();

        if (gridOld[offsetZ + dimX + 1] == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        Coord<3> actualDim = dim - Coord<3>(2, 2, 2);
        double updates = 1.0 * maxT * actualDim.prod();
        double seconds = (tEnd - tBegin) * 10e-6;
        double gLUPS = 10e-9 * updates / seconds;

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
    NoOpInitializer(
        const Coord<3>& dimensions,
        const unsigned& steps) :
        SimpleInitializer<CELL>(dimensions, steps)
    {}
    
    virtual void grid(GridBase<CELL, CELL::Topology::DIMENSIONS> *target) 
    {}
};

class JacobiCellClassic
{
public:
    typedef Stencils::VonNeumann<3, 1> Stencil;
    typedef Topologies::Cube<3>::Topology Topology;
    class API : public CellAPITraits::Base
    {};

    static int nanoSteps()
    {
        return 1;
    }

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

class Jacobi3DClassic
{
public:
    std::string order()
    {
        return "CPU";
    }

    std::string family()
    {
        return "Jacobi3D";
    }

    std::string species()
    {
        return "bronze";
    }

    double performance(const Coord<3>& dim)
    {
        int maxT = 5;
        SerialSimulator<JacobiCellClassic> sim(
            new NoOpInitializer<JacobiCellClassic>(dim, maxT));

        long long tBegin= Chronometer::timeUSec();
        sim.run();
        long long tEnd = Chronometer::timeUSec();

        if (sim.getGrid()->at(Coord<3>(1, 1, 1)).temp == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        double updates = 1.0 * maxT * dim.prod();
        double seconds = (tEnd - tBegin) * 10e-6;
        double gLUPS = 10e-9 * updates / seconds;

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
    typedef Stencils::VonNeumann<3, 1> Stencil;
    typedef Topologies::Cube<3>::Topology Topology;
    class API : public CellAPITraits::Fixed
    {};

    JacobiCellFixedHood(double t = 0) :
        temp(t)
    {}

    static int nanoSteps()
    {
        return 1;
    }

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

class Jacobi3DFixedHood
{
public:
    std::string order()
    {
        return "CPU";
    }

    std::string family()
    {
        return "Jacobi3D";
    }

    std::string species()
    {
        return "silver";
    }

    double performance(const Coord<3>& dim)
    {
        int maxT = 20;

        SerialSimulator<JacobiCellFixedHood> sim(
            new NoOpInitializer<JacobiCellFixedHood>(dim, maxT));

        long long tBegin= Chronometer::timeUSec();
        sim.run();
        long long tEnd = Chronometer::timeUSec();

        if (sim.getGrid()->at(Coord<3>(1, 1, 1)).temp == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        double updates = 1.0 * maxT * dim.prod();
        double seconds = (tEnd - tBegin) * 10e-6;
        double gLUPS = 10e-9 * updates / seconds;

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
    typedef Stencils::VonNeumann<3, 1> Stencil;
    typedef Topologies::Cube<3>::Topology Topology;
  
    class API : public CellAPITraits::Fixed, public CellAPITraits::Line
    {};

    JacobiCellStreakUpdate(double t = 0) :
        temp(t)
    {}

    static int nanoSteps()
    {
        return 1;
    }

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, int /* nanoStep */)
    {
        temp = (hood[FixedCoord<0,  0, -1>()].temp +
                hood[FixedCoord<0, -1,  0>()].temp +
                hood[FixedCoord<1,  0,  0>()].temp +
                hood[FixedCoord<0,  0,  0>()].temp +
                hood[FixedCoord<1,  0,  0>()].temp +
                hood[FixedCoord<0,  1,  0>()].temp +
                hood[FixedCoord<0,  0,  1>()].temp) * (1.0 / 7.0);
    }

    template<typename NEIGHBORHOOD>
    static void updateLine(JacobiCellStreakUpdate *target, long *x, long endX, const NEIGHBORHOOD& hood, int /* nanoStep */)
    {
        if (((*x) % 2) == 1) {
            target[*x].update(hood, 0);
            ++(*x);
        }

        __m128d oneSeventh = _mm_set_pd(1.0/7.0, 1.0/7.0);

        PentaM128 same;
        // PentaM128 odds;

        same.a = _mm_load_pd( &hood[FixedCoord< 0, 0, 0>()].temp);
        __m128d odds0 = _mm_loadu_pd(&hood[FixedCoord<-1, 0, 0>()].temp);

        for (; (*x) < (endX - 7); (*x) += 8) {
            load(&same, hood, FixedCoord<0, 0, 0>());
            // __m128d same2 = _mm_load_pd(&hood[FixedCoord< 2, 0, 0>()].temp);
            // __m128d same3 = _mm_load_pd(&hood[FixedCoord< 4, 0, 0>()].temp);
            // __m128d same4 = _mm_load_pd(&hood[FixedCoord< 6, 0, 0>()].temp);
            // __m128d same5 = _mm_load_pd(&hood[FixedCoord< 8, 0, 0>()].temp);

            // shuffle values obtain left/right neighbors
            __m128d odds1 = _mm_shuffle_pd(same.a, same.b, (1 << 0) | (0 << 2));
            __m128d odds2 = _mm_shuffle_pd(same.b, same.c, (1 << 0) | (0 << 2));
            __m128d odds3 = _mm_shuffle_pd(same.c, same.d, (1 << 0) | (0 << 2));
            __m128d odds4 = _mm_shuffle_pd(same.d, same.e, (1 << 0) | (0 << 2));

            // load south neighbors
            QuadM128 buf;
            load(&buf, hood, FixedCoord<0, 0, -1>());
            // __m128d buf0 =  load<0>(hood, FixedCoord< 0, 0, -1>());
            // __m128d buf1 =  load<2>(hood, FixedCoord< 0, 0, -1>());
            // __m128d buf2 =  load<4>(hood, FixedCoord< 0, 0, -1>());
            // __m128d buf3 =  load<6>(hood, FixedCoord< 0, 0, -1>());

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
            odds0 = load<0>(hood, FixedCoord< 0, -1, 0>());
            odds1 = load<2>(hood, FixedCoord< 0, -1, 0>());
            odds2 = load<4>(hood, FixedCoord< 0, -1, 0>());
            odds3 = load<6>(hood, FixedCoord< 0, -1, 0>());

            // add south neighbors
            same.a = _mm_add_pd(same.a, buf.a);
            same.b = _mm_add_pd(same.b, buf.b);
            same.c = _mm_add_pd(same.c, buf.c);
            same.d = _mm_add_pd(same.d, buf.d);

            // load bottom neighbors
            load(&buf, hood, FixedCoord<0, 1, 0>());
            // buf0 =  load<0>(hood, FixedCoord< 0, 1, 0>());
            // buf1 =  load<2>(hood, FixedCoord< 0, 1, 0>());
            // buf2 =  load<4>(hood, FixedCoord< 0, 1, 0>());
            // buf3 =  load<6>(hood, FixedCoord< 0, 1, 0>());

            // add top neighbors
            same.a = _mm_add_pd(same.a, odds0);
            same.b = _mm_add_pd(same.b, odds1);
            same.c = _mm_add_pd(same.c, odds2);
            same.d = _mm_add_pd(same.d, odds3);

            // load north neighbors
            odds0 = load<0>(hood, FixedCoord< 0, 0, 1>());
            odds1 = load<2>(hood, FixedCoord< 0, 0, 1>());
            odds2 = load<4>(hood, FixedCoord< 0, 0, 1>());
            odds3 = load<6>(hood, FixedCoord< 0, 0, 1>());

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
            _mm_store_pd(&target[*x + 0].temp, same.a);
            _mm_store_pd(&target[*x + 2].temp, same.b);
            _mm_store_pd(&target[*x + 4].temp, same.c);
            _mm_store_pd(&target[*x + 6].temp, same.d);

            odds0 = odds4;
            same.a = same.e;
        }

        for (; *x < endX; ++(*x)) {
            target[*x].update(hood, 0);
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
        return load<OFFSET>(&hood[coord].temp, hood.arity(coord));
    }

    template<int OFFSET>
    static __m128d load(const double *p, VectorArithmetics::Vector)
    {
        return _mm_load_pd(p + OFFSET);
    }

    template<int OFFSET>
    static __m128d load(const double *p, VectorArithmetics::Scalar)
    {
        return _mm_set_pd(*p, *p);
    }

    double temp;
};

class Jacobi3DStreakUpdate
{
public:
    std::string order()
    {
        return "CPU";
    }

    std::string family()
    {
        return "Jacobi3D";
    }

    std::string species()
    {
        return "gold";
    }

    double performance(const Coord<3>& dim)
    {
        typedef Grid<JacobiCellStreakUpdate, JacobiCellStreakUpdate::Topology> GridType;
        GridType gridA(dim, 1.0);
        GridType gridB(dim, 2.0);
        GridType *gridOld = &gridA;
        GridType *gridNew = &gridB;

        int maxT = 20;

        CoordBox<3> gridBox = gridA.boundingBox();
        CoordBox<3> lineStarts = gridA.boundingBox();
        lineStarts.dimensions.x() = 1;

        long long tBegin= Chronometer::timeUSec();

        for (int t = 0; t < maxT; ++t) {
            for (CoordBox<3>::Iterator i = lineStarts.begin();
                 i != lineStarts.end();
                 ++i) {
                Streak<3> streak(*i, dim.x());
                const JacobiCellStreakUpdate *pointers[JacobiCellStreakUpdate::Stencil::VOLUME];
                LinePointerAssembly<JacobiCellStreakUpdate::Stencil>()(pointers, streak, *gridOld);
                LinePointerUpdateFunctor<JacobiCellStreakUpdate>()(
                    streak, gridBox, pointers, &(*gridNew)[streak.origin], 0);
            }

            std::swap(gridOld, gridNew);
        }

        long long tEnd = Chronometer::timeUSec();

        if (gridA[Coord<3>(1, 1, 1)].temp == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        double updates = 1.0 * maxT * dim.prod();
        double seconds = (tEnd - tBegin) * 10e-6;
        double gLUPS = 10e-9 * updates / seconds;

        return gLUPS;
    }

    std::string unit()
    {
        return "GLUPS";
    }
};

class Jacobi3DStreakUpdateFunctor
{
public:
    std::string order()
    {
        return "CPU";
    }

    std::string family()
    {
        return "Jacobi3D";
    }

    std::string species()
    {
        return "platinum";
    }

    double performance(const Coord<3>& dim)
    {
        int maxT = 20;
        SerialSimulator<JacobiCellStreakUpdate> sim(
            new NoOpInitializer<JacobiCellStreakUpdate>(dim, maxT));

        long long tBegin= Chronometer::timeUSec();
        sim.run();
        long long tEnd = Chronometer::timeUSec();

        if (sim.getGrid()->at(Coord<3>(1, 1, 1)).temp == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        double updates = 1.0 * maxT * dim.prod();
        double seconds = (tEnd - tBegin) * 10e-6;
        double gLUPS = 10e-9 * updates / seconds;

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
    typedef Stencils::Moore<3, 1> Stencil;
    typedef Topologies::Cube<3>::Topology Topology;

    class API : public CellAPITraits::Base
    {};

    enum State {LIQUID, WEST_NOSLIP, EAST_NOSLIP, TOP, BOTTOM, NORTH_ACC, SOUTH_NOSLIP};

    static inline unsigned nanoSteps() 
    { 
        return 1; 
    }

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

    inline explicit LBMCell(const double& v=1.0, const State& s=LIQUID) :
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

class LBMClassic
{
public:
    std::string order()
    {
        return "CPU";
    }

    std::string family()
    {
        return "LBM";
    }

    std::string species()
    {
        return "bronze";
    }

    double performance(const Coord<3>& dim)
    {
        int maxT = 5;
        SerialSimulator<LBMCell> sim(
            new NoOpInitializer<LBMCell>(dim, maxT));

        long long tBegin= Chronometer::timeUSec();
        sim.run();
        long long tEnd = Chronometer::timeUSec();

        if (sim.getGrid()->at(Coord<3>(1, 1, 1)).density == 4711) {
            std::cout << "this statement just serves to prevent the compiler from"
                      << "optimizing away the loops above\n";
        }

        double updates = 1.0 * maxT * dim.prod();
        double seconds = (tEnd - tBegin) * 10e-6;
        double gLUPS = 10e-9 * updates / seconds;

        return gLUPS;
    }

    std::string unit()
    {
        return "GLUPS";
    }
};

template<class BENCHMARK>
void evaluate(BENCHMARK benchmark, const Coord<3>& dim)
{
    boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
    std::stringstream buf;
    buf << now;
    std::string nowString = buf.str();
    nowString.resize(20);

    int hostnameLength = 2048;
    std::string hostname(hostnameLength, ' ');
    gethostname(&hostname[0], hostnameLength);
    int actualLength = 0;
    for (int i = 0; i < hostnameLength; ++i) {
        if (hostname[i] == 0) {
            actualLength = i;
        }
    }
    hostname.resize(actualLength);

    FILE *output = popen("cat /proc/cpuinfo | grep 'model name' | head -1 | cut -c 14-", "r");
    int idLength = 2048;
    std::string cpuID(idLength, ' ');
    idLength = fread(&cpuID[0], 1, idLength, output);
    cpuID.resize(idLength - 1);
    pclose(output);
    

    std::cout << std::setiosflags(std::ios::left);
    std::cout << std::setw(18) << revision << "; " 
              << nowString << " ; " 
              << std::setw(16) << hostname << "; " 
              << std::setw(48) << cpuID << "; " 
              << std::setw( 8) << benchmark.order() <<  "; " 
              << std::setw(16) << benchmark.family() <<  "; " 
              << std::setw( 8) << benchmark.species() <<  "; " 
              << std::setw(24) << dim <<  "; " 
              << std::setw(12) << benchmark.performance(dim) <<  "; " 
              << std::setw( 8) << benchmark.unit() <<  "\n";
}

#ifdef LIBGEODECOMP_FEATURE_CUDA
void cudaTests(std::string revision, bool quick, int cudaDevice);
#endif

int main(int argc, char **argv)
{
    if ((argc < 3) || (argc > 4)) {
        std::cerr << "usage: " << argv[0] << "[-q,--quick] REVISION CUDA_DEVICE\n";
        return 1;
    }

    bool quick = false;
    int revIndex = 1;
    if (argc == 4) {
        if ((std::string(argv[1]) == "-q") ||
            (std::string(argv[1]) == "--quick")) {
            quick = true;
        }
        revIndex = 2;
    }
    revision = argv[revIndex];
    std::stringstream s;
    s << argv[revIndex + 1];
    int cudaDevice;
    s >> cudaDevice;

    std::cout << "#rev              ; date                 ; host            ; device                                          ; order   ; family          ; species ; dimensions              ; perf        ; unit\n";

    evaluate(RegionCount(), Coord<3>( 128,  128,  128));
    evaluate(RegionCount(), Coord<3>( 512,  512,  512));
    evaluate(RegionCount(), Coord<3>(2048, 2048, 2048));

    evaluate(RegionInsert(), Coord<3>( 128,  128,  128));
    evaluate(RegionInsert(), Coord<3>( 512,  512,  512));
    evaluate(RegionInsert(), Coord<3>(2048, 2048, 2048));

    evaluate(RegionIntersect(), Coord<3>( 128,  128,  128));
    evaluate(RegionIntersect(), Coord<3>( 512,  512,  512));
    evaluate(RegionIntersect(), Coord<3>(2048, 2048, 2048));

    evaluate(CoordEnumerationVanilla(), Coord<3>( 128,  128,  128));
    evaluate(CoordEnumerationVanilla(), Coord<3>( 512,  512,  512));
    evaluate(CoordEnumerationVanilla(), Coord<3>(2048, 2048, 2048));

    evaluate(CoordEnumerationBronze(), Coord<3>( 128,  128,  128));
    evaluate(CoordEnumerationBronze(), Coord<3>( 512,  512,  512));
    evaluate(CoordEnumerationBronze(), Coord<3>(2048, 2048, 2048));

    evaluate(CoordEnumerationGold(), Coord<3>( 128,  128,  128));
    evaluate(CoordEnumerationGold(), Coord<3>( 512,  512,  512));
    evaluate(CoordEnumerationGold(), Coord<3>(2048, 2048, 2048));

    SuperVector<Coord<3> > sizes;
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

    for (int i = 0; i < sizes.size(); ++i) {
        evaluate(Jacobi3DVanilla(), sizes[i]);
    }

    for (int i = 0; i < sizes.size(); ++i) {
        evaluate(Jacobi3DSSE(), sizes[i]);
    }

    for (int i = 0; i < sizes.size(); ++i) {
        evaluate(Jacobi3DClassic(), sizes[i]);
    }

    for (int i = 0; i < sizes.size(); ++i) {
        evaluate(Jacobi3DFixedHood(), sizes[i]);
    }

    for (int i = 0; i < sizes.size(); ++i) {
        evaluate(Jacobi3DStreakUpdate(), sizes[i]);
    }

    for (int i = 0; i < sizes.size(); ++i) {
        evaluate(Jacobi3DStreakUpdateFunctor(), sizes[i]);
    }

    sizes.clear();

    sizes << Coord<3>(22, 22, 22)
          << Coord<3>(64, 64, 64)
          << Coord<3>(68, 68, 68)
          << Coord<3>(106, 106, 106)
          << Coord<3>(128, 128, 128);

    for (int i = 0; i < sizes.size(); ++i) {
        evaluate(LBMClassic(), sizes[i]);
    }

#ifdef LIBGEODECOMP_FEATURE_CUDA
    // cudaTests(revision, quick, cudaDevice);
#endif

    return 0;
}
