#include <boost/date_time/posix_time/posix_time.hpp>
#include <emmintrin.h>
#include <iomanip>
#include <iostream>
#include <libgeodecomp/misc/apis.h>
#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/misc/linepointerassembly.h>
#include <libgeodecomp/misc/linepointerupdatefunctor.h>
#include <libgeodecomp/misc/region.h>
#include <libgeodecomp/misc/stencils.h>
#include <stdio.h>

using namespace LibGeoDecomp;

std::string revision;

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
            same1 = _mm_add_pd(same1, oneSeventh);
            same2 = _mm_add_pd(same2, oneSeventh);
            same3 = _mm_add_pd(same3, oneSeventh);
            same4 = _mm_add_pd(same4, oneSeventh);

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

class JacobiCellClassic
{
public:
    typedef Stencils::Moore<3, 1> Stencil;
    typedef Topologies::Cube<3>::Topology Topology;
    class API : public APIs::Base
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
                hood[Coord<3>( 1,  0,  1)].temp) * (1.0 / 7.0);
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
        typedef Grid<JacobiCellClassic, JacobiCellClassic::Topology> GridType;
        GridType gridA(dim, 1.0);
        GridType gridB(dim, 2.0);
        GridType *gridOld = &gridA;
        GridType *gridNew = &gridB;

        int maxT = 5;
        CoordBox<3> coords = gridOld->boundingBox();

        long long tBegin= Chronometer::timeUSec();

        for (int t = 0; t < maxT; ++t) {
            for (typename CoordBox<3>::Iterator i = coords.begin();
                 i != coords.end();
                 ++i) {
                CoordMap<JacobiCellClassic, GridType> neighborhood = 
                gridOld->getNeighborhood(*i);
                (*gridNew)[*i].update(neighborhood, 0);
                // Streak<3> streak(*i, dim.x());
                // TestCellType *pointers[JacobiCellClassic::Stencil::VOLUME];
                // LinePointerAssembly<JacobiCellClassic::Stencil>()(pointers, streak, gridA);
                // LinePointerUpdateFunctor<TestCellType>()(
                //     streak, gridBox, pointers, &gridNew[streak.origin], s);
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

class JacobiCellFixedHood
{
public:
    typedef Stencils::Moore<3, 1> Stencil;
    typedef Topologies::Cube<3>::Topology Topology;
    class API : public APIs::Fixed
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
        temp = (hood[FixedCoord<0,  0, -1>()].temp +
                hood[FixedCoord<0, -1,  0>()].temp +
                hood[FixedCoord<1,  0,  0>()].temp +
                hood[FixedCoord<0,  0,  0>()].temp +
                hood[FixedCoord<1,  0,  0>()].temp +
                hood[FixedCoord<0,  1,  0>()].temp +
                hood[FixedCoord<1,  0,  1>()].temp) * (1.0 / 7.0);
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
        return "gold";
    }

    double performance(const Coord<3>& dim)
    {
        typedef Grid<JacobiCellFixedHood, JacobiCellFixedHood::Topology> GridType;
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
            for (typename CoordBox<3>::Iterator i = lineStarts.begin();
                 i != lineStarts.end();
                 ++i) {
                Streak<3> streak(*i, dim.x());
                JacobiCellFixedHood *pointers[JacobiCellFixedHood::Stencil::VOLUME];
                LinePointerAssembly<JacobiCellFixedHood::Stencil>()(pointers, streak, gridA);
                LinePointerUpdateFunctor<JacobiCellFixedHood>()(
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

class JacobiCellStreakUpdate
{
public:
    typedef Stencils::Moore<3, 1> Stencil;
    typedef Topologies::Torus<3>::Topology Topology;
    class API : public APIs::Fixed, public APIs::Line
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
                hood[FixedCoord<1,  0,  1>()].temp) * (1.0 / 7.0);
    }

    template<typename NEIGHBORHOOD>
    static void updateLine(JacobiCellStreakUpdate *target, long *x, long endX, const NEIGHBORHOOD& hood, int /* nanoStep */)
    {
        if (((*x) % 2) == 1) {
            target[*x].update(hood, 0);
            ++(*x);
        }

        __m128d oneSeventh = _mm_set_pd(1.0/7.0, 1.0/7.0);
        __m128d same1 = _mm_load_pd( &hood[FixedCoord< 0, 0, 0>()].temp);
        __m128d odds0 = _mm_loadu_pd(&hood[FixedCoord<-1, 0, 0>()].temp);

        for (; (*x) < (endX - 7); (*x) += 8) {
            __m128d same2 = _mm_load_pd(&hood[FixedCoord< 2, 0, 0>()].temp);
            __m128d same3 = _mm_load_pd(&hood[FixedCoord< 4, 0, 0>()].temp);
            __m128d same4 = _mm_load_pd(&hood[FixedCoord< 6, 0, 0>()].temp);
            __m128d same5 = _mm_load_pd(&hood[FixedCoord< 8, 0, 0>()].temp);

            // shuffle values obtain left/right neighbors
            __m128d odds1 = _mm_shuffle_pd(same1, same2, (1 << 0) | (0 << 2));
            __m128d odds2 = _mm_shuffle_pd(same2, same3, (1 << 0) | (0 << 2));
            __m128d odds3 = _mm_shuffle_pd(same3, same4, (1 << 0) | (0 << 2));
            __m128d odds4 = _mm_shuffle_pd(same4, same5, (1 << 0) | (0 << 2));

            // load south neighbors
            __m128d buf0 =  _mm_load_pd(&hood[FixedCoord< 0, 0, -1>()].temp);
            __m128d buf1 =  _mm_load_pd(&hood[FixedCoord< 2, 0, -1>()].temp);
            __m128d buf2 =  _mm_load_pd(&hood[FixedCoord< 4, 0, -1>()].temp);
            __m128d buf3 =  _mm_load_pd(&hood[FixedCoord< 6, 0, -1>()].temp);

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
            odds0 = _mm_load_pd(&hood[FixedCoord< 0, -1, 0>()].temp);
            odds1 = _mm_load_pd(&hood[FixedCoord< 2, -1, 0>()].temp);
            odds2 = _mm_load_pd(&hood[FixedCoord< 4, -1, 0>()].temp);
            odds3 = _mm_load_pd(&hood[FixedCoord< 6, -1, 0>()].temp);

            // add south neighbors
            same1 = _mm_add_pd(same1, buf0);
            same2 = _mm_add_pd(same2, buf1);
            same3 = _mm_add_pd(same3, buf2);
            same4 = _mm_add_pd(same4, buf3);

            // load bottom neighbors
            buf0 =  _mm_load_pd(&hood[FixedCoord< 0, 1, 0>()].temp);
            buf1 =  _mm_load_pd(&hood[FixedCoord< 2, 1, 0>()].temp);
            buf2 =  _mm_load_pd(&hood[FixedCoord< 4, 1, 0>()].temp);
            buf3 =  _mm_load_pd(&hood[FixedCoord< 6, 1, 0>()].temp);

            // add top neighbors
            same1 = _mm_add_pd(same1, odds0);
            same2 = _mm_add_pd(same2, odds1);
            same3 = _mm_add_pd(same3, odds2);
            same4 = _mm_add_pd(same4, odds3);

            // load north neighbors
            odds0 = _mm_load_pd(&hood[FixedCoord< 0, 0, 1>()].temp);
            odds1 = _mm_load_pd(&hood[FixedCoord< 2, 0, 1>()].temp);
            odds2 = _mm_load_pd(&hood[FixedCoord< 4, 0, 1>()].temp);
            odds3 = _mm_load_pd(&hood[FixedCoord< 6, 0, 1>()].temp);

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
            same1 = _mm_add_pd(same1, oneSeventh);
            same2 = _mm_add_pd(same2, oneSeventh);
            same3 = _mm_add_pd(same3, oneSeventh);
            same4 = _mm_add_pd(same4, oneSeventh);

            // store results
            _mm_store_pd(&target[*x + 0].temp, same1);
            _mm_store_pd(&target[*x + 2].temp, same2);
            _mm_store_pd(&target[*x + 4].temp, same3);
            _mm_store_pd(&target[*x + 6].temp, same4);

            odds0 = odds4;
            same1 = same5;
        }

        for (; *x < endX; ++(*x)) {
            target[*x].update(hood, 0);
        }
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
        return "platinum";
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
            for (typename CoordBox<3>::Iterator i = lineStarts.begin();
                 i != lineStarts.end();
                 ++i) {
                Streak<3> streak(*i, dim.x());
                JacobiCellStreakUpdate *pointers[JacobiCellStreakUpdate::Stencil::VOLUME];
                LinePointerAssembly<JacobiCellStreakUpdate::Stencil>()(pointers, streak, gridA);
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

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " REVISION\n";
        return 1;
    }

    revision = argv[1];

    std::cout << "#rev              ; date                 ; host            ; device                                          ; order   ; family          ; species ; dimensions              ; perf        ; unit\n";

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

    return 0;
}
