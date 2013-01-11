#include <cmath>
#include <emmintrin.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>

using namespace LibGeoDecomp;

class JacobiCellSimple
{
public:
    typedef Stencils::VonNeumann<3, 1> Stencil;
    typedef Topologies::Torus<3>::Topology Topology;
  
    class API : public CellAPITraits::Fixed
    {};

    JacobiCellSimple(double t = 0) :
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

    double temp;
};

class JacobiCellMagic
{
public:
    typedef Stencils::VonNeumann<3, 1> Stencil;
    typedef Topologies::Torus<3>::Topology Topology;
  
    class API : public CellAPITraits::Fixed, public CellAPITraits::Line
    {};

    JacobiCellMagic(double t = 0) :
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
    static void updateLine(JacobiCellMagic *target, long *x, long endX, const NEIGHBORHOOD& hood, int /* nanoStep */)
    {
        for (; *x < endX; ++*x) {
            target[*x].update(hood, 0);
        }
    }

    double temp;
};

class JacobiCellStraightforward
{
public:
    typedef Stencils::VonNeumann<3, 1> Stencil;
    typedef Topologies::Torus<3>::Topology Topology;
  
    class API : public CellAPITraits::Fixed, public CellAPITraits::Line
    {};

    JacobiCellStraightforward(double t = 0) :
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
    static void updateLine(JacobiCellStraightforward *target, long *x, long endX, const NEIGHBORHOOD& hood, int /* nanoStep */)
    {
        if (((*x) % 2) == 1) {
            target[*x].update(hood, 0);
            ++(*x);
        }

        __m128d oneSeventh = _mm_set_pd(1.0/7.0, 1.0/7.0);

        for (; (*x) < (endX - 8); (*x) += 8) {
            __m128d accu0 = _mm_load_pd(&hood[FixedCoord< 0, 0, 0>()].temp);
            __m128d accu1 = _mm_load_pd(&hood[FixedCoord< 2, 0, 0>()].temp);
            __m128d accu2 = _mm_load_pd(&hood[FixedCoord< 4, 0, 0>()].temp);
            __m128d accu3 = _mm_load_pd(&hood[FixedCoord< 6, 0, 0>()].temp);

            __m128d buff0 = _mm_loadu_pd(&hood[FixedCoord<-1, 0, 0>()].temp);
            __m128d buff1 = _mm_loadu_pd(&hood[FixedCoord< 1, 0, 0>()].temp);
            __m128d buff2 = _mm_loadu_pd(&hood[FixedCoord< 3, 0, 0>()].temp);
            __m128d buff3 = _mm_loadu_pd(&hood[FixedCoord< 5, 0, 0>()].temp);
            accu0 = _mm_add_pd(accu0, buff0);
            accu1 = _mm_add_pd(accu1, buff1);
            accu2 = _mm_add_pd(accu2, buff2);
            accu3 = _mm_add_pd(accu3, buff3);

            buff0 = _mm_loadu_pd(&hood[FixedCoord< 1, 0, 0>()].temp);
            buff1 = _mm_loadu_pd(&hood[FixedCoord< 3, 0, 0>()].temp);
            buff2 = _mm_loadu_pd(&hood[FixedCoord< 5, 0, 0>()].temp);
            buff3 = _mm_loadu_pd(&hood[FixedCoord< 7, 0, 0>()].temp);
            accu0 = _mm_add_pd(accu0, buff0);
            accu1 = _mm_add_pd(accu1, buff1);
            accu2 = _mm_add_pd(accu2, buff2);
            accu3 = _mm_add_pd(accu3, buff3);

            buff0 = _mm_load_pd(&hood[FixedCoord< 0, -1, 0>()].temp);
            buff1 = _mm_load_pd(&hood[FixedCoord< 2, -1, 0>()].temp);
            buff2 = _mm_load_pd(&hood[FixedCoord< 4, -1, 0>()].temp);
            buff3 = _mm_load_pd(&hood[FixedCoord< 6, -1, 0>()].temp);
            accu0 = _mm_add_pd(accu0, buff0);
            accu1 = _mm_add_pd(accu1, buff1);
            accu2 = _mm_add_pd(accu2, buff2);
            accu3 = _mm_add_pd(accu3, buff3);

            buff0 = _mm_load_pd(&hood[FixedCoord< 0,  1, 0>()].temp);
            buff1 = _mm_load_pd(&hood[FixedCoord< 2,  1, 0>()].temp);
            buff2 = _mm_load_pd(&hood[FixedCoord< 4,  1, 0>()].temp);
            buff3 = _mm_load_pd(&hood[FixedCoord< 6,  1, 0>()].temp);
            accu0 = _mm_add_pd(accu0, buff0);
            accu1 = _mm_add_pd(accu1, buff1);
            accu2 = _mm_add_pd(accu2, buff2);
            accu3 = _mm_add_pd(accu3, buff3);

            buff0 = _mm_load_pd(&hood[FixedCoord< 0, 0, -1>()].temp);
            buff1 = _mm_load_pd(&hood[FixedCoord< 2, 0, -1>()].temp);
            buff2 = _mm_load_pd(&hood[FixedCoord< 4, 0, -1>()].temp);
            buff3 = _mm_load_pd(&hood[FixedCoord< 6, 0, -1>()].temp);
            accu0 = _mm_add_pd(accu0, buff0);
            accu1 = _mm_add_pd(accu1, buff1);
            accu2 = _mm_add_pd(accu2, buff2);
            accu3 = _mm_add_pd(accu3, buff3);

            buff0 = _mm_load_pd(&hood[FixedCoord< 0, 0, 1>()].temp);
            buff1 = _mm_load_pd(&hood[FixedCoord< 2, 0, 1>()].temp);
            buff2 = _mm_load_pd(&hood[FixedCoord< 4, 0, 1>()].temp);
            buff3 = _mm_load_pd(&hood[FixedCoord< 6, 0, 1>()].temp);
            accu0 = _mm_add_pd(accu0, buff0);
            accu1 = _mm_add_pd(accu1, buff1);
            accu2 = _mm_add_pd(accu2, buff2);
            accu3 = _mm_add_pd(accu3, buff3);

            accu0 = _mm_mul_pd(accu0, oneSeventh);
            accu1 = _mm_mul_pd(accu1, oneSeventh);
            accu2 = _mm_mul_pd(accu2, oneSeventh);
            accu3 = _mm_mul_pd(accu3, oneSeventh);

            _mm_store_pd(&target[*x + 0].temp, accu0);
            _mm_store_pd(&target[*x + 2].temp, accu1);
            _mm_store_pd(&target[*x + 4].temp, accu2);
            _mm_store_pd(&target[*x + 6].temp, accu3);
        }

        for (; *x < endX; ++(*x)) {
            target[*x].update(hood, 0);
        }


    }

    double temp;
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
    typedef Topologies::Torus<3>::Topology Topology;
  
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

        for (; (*x) < (endX - 8); (*x) += 8) {
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

template<class CELL>
class MonoInitializer : public SimpleInitializer<CELL>
{
public:
    MonoInitializer(const Coord<3>& dim, int steps) : SimpleInitializer<CELL>(dim, steps)
    {}

    virtual void grid(GridBase<CELL, 3> *ret)
    {
        CoordBox<3> box = ret->boundingBox();
        for (CoordBox<3>::Iterator i = box.begin(); i != box.end(); ++i) {
            ret->at(*i) = CELL(1.0);
        }
        ret->atEdge() = CELL(0.0);
    }
};

template<class CELL>
double singleBenchmark(Coord<3> dim)
{
    int repeats = 50.0 * 1000 * 1000 / dim.prod();
    repeats = std::max(repeats, 10);

    SerialSimulator<CELL> sim(
        new MonoInitializer<CELL>(dim, repeats));
    // sim.addWriter(new TracingWriter<CELL>(500, repeats));

    long long tBegin= Chronometer::timeUSec();
    sim.run();
    long long tEnd = Chronometer::timeUSec();

    if (sim.getGrid()->at(Coord<3>(1, 1, 1)).temp == 4711) {
        std::cout << "this statement just serves to prevent the compiler from"
                  << "optimizing away the loops above\n";
    }
    
    double updates = 1.0 * repeats * (dim - Coord<3>::diagonal(2)).prod();
    double seconds = (tEnd - tBegin) * 10e-6;
    double glups = 10e-9 * updates / seconds;
    
    return glups;
}

template<class CELL>
void benchmark(std::string name)
{
    std::cout << "Benchmarking " << name << "\n";

    std::stringstream fileName;
    fileName << "data." << name << ".txt";
    std::ofstream outfile(fileName.str().c_str());
    outfile << "# BENCHMARK_ID MAXTRIX_SIZE GLUPS\n";
    
    int maxI = 24;

    for (int i = 0; i < maxI; ++i) {
        int dim = std::pow(2, 4 + 0.25 * i);
        if (dim % 2) {
            ++dim;
        }

        double glups = singleBenchmark<CELL>(Coord<3>::diagonal(dim));
        outfile << dim << " " << glups << "\n";

        int percent = 100 * i / (maxI - 1);
        std::cout << std::setw(3) << percent << "%, dim = " << std::setw(3) << dim << ", " << std::setw(8) << glups << " GLUPS ";
        for (int dots = 0; dots < i; ++dots) {
            std::cout << ".";
        }
        std::cout.flush();
        std::cout << "\r";
    }

    std::cout << "\n";
}

int main(int argc, char **argv)
{
    benchmark<JacobiCellSimple         >("JacobiCellSimple");
    benchmark<JacobiCellMagic          >("JacobiCellMagic");
    benchmark<JacobiCellStraightforward>("JacobiCellStraightforward");
    benchmark<JacobiCellStreakUpdate   >("JacobiCellStreakUpdate");
    return 0;
}
