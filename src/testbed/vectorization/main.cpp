#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/misc/chronometer.h>
#include <libgeodecomp/parallelization/serialsimulator.h>

#include <cmath>
#include <emmintrin.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace LibGeoDecomp;

class JacobiCellSimple
{
public:
    class API :
        public APITraits::HasFixedCoordsOnlyUpdate,
        public APITraits::HasStencil<Stencils::VonNeumann<3, 1> >,
        public APITraits::HasTorusTopology<3>
    {};

    JacobiCellSimple(double t = 0) :
        temp(t)
    {}

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

    const double& read() const
    {
        return temp;
    }

    double temp;
};

class JacobiCellMagic
{
public:
    class API :
        public APITraits::HasFixedCoordsOnlyUpdate,
        public APITraits::HasUpdateLineX,
        public APITraits::HasStencil<Stencils::VonNeumann<3, 1> >,
        public APITraits::HasTorusTopology<3>
    {};

    JacobiCellMagic(double t = 0) :
        temp(t)
    {}

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
    static void updateLineX(JacobiCellMagic *target, long *x, long endX, const NEIGHBORHOOD& hood, int /* nanoStep */)
    {
        for (; *x < endX; ++*x) {
            target[*x].update(hood, 0);
        }
    }

    const double& read() const
    {
        return temp;
    }

    double temp;
};

class JacobiCellStraightforward
{
public:
    class API :
        public APITraits::HasFixedCoordsOnlyUpdate,
        public APITraits::HasUpdateLineX,
        public APITraits::HasStencil<Stencils::VonNeumann<3, 1> >,
        public APITraits::HasTorusTopology<3>
    {};

    JacobiCellStraightforward(double t = 0) :
        temp(t)
    {}

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
    static void updateLineX(JacobiCellStraightforward *target, long *x, long endX, const NEIGHBORHOOD& hood, int /* nanoStep */)
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

    const double& read() const
    {
        return temp;
    }

    double temp;
};

class JacobiCellStraightforwardNT
{
public:
    class API :
        public APITraits::HasFixedCoordsOnlyUpdate,
        public APITraits::HasUpdateLineX,
        public APITraits::HasStencil<Stencils::VonNeumann<3, 1> >,
        public APITraits::HasTorusTopology<3>
    {};

    JacobiCellStraightforwardNT(double t = 0) :
        temp(t)
    {}

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
    static void updateLineX(JacobiCellStraightforwardNT *target, long *x, long endX, const NEIGHBORHOOD& hood, int /* nanoStep */)
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

            _mm_stream_pd(&target[*x + 0].temp, accu0);
            _mm_stream_pd(&target[*x + 2].temp, accu1);
            _mm_stream_pd(&target[*x + 4].temp, accu2);
            _mm_stream_pd(&target[*x + 6].temp, accu3);
        }

        for (; *x < endX; ++(*x)) {
            target[*x].update(hood, 0);
        }


    }

    const double& read() const
    {
        return temp;
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
    class API :
        public APITraits::HasFixedCoordsOnlyUpdate,
        public APITraits::HasUpdateLineX,
        public APITraits::HasStencil<Stencils::VonNeumann<3, 1> >,
        public APITraits::HasTorusTopology<3>
    {};

    JacobiCellStreakUpdate(double t = 0) :
        temp(t)
    {}

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
    static void updateLineX(JacobiCellStreakUpdate *target, long *x, long endX, const NEIGHBORHOOD& hood, int /* nanoStep */)
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

    const double& read() const
    {
        return temp;
    }

    double temp;
};

void store(double *a, double v)
{
    *a = v;
}

template<typename VEC>
void store(double *a, VEC v)
{
    v.store(a);
}

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

// this class is a quick helper to allow us to discover the number of
// elements in a given (short vector) float type:
template<typename T>
class ArityHelper;

template<>
class ArityHelper<double>
{
public:
    static const int VALUE = 1;
};

template<>
class ArityHelper<ShortVec4xSSE>
{
public:
    static const int VALUE = 8;
};

template<typename DOUBLE>
class LBMSoACell
{
public:
    typedef DOUBLE Double;

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
        ACCESSOR1 hoodOld, int *indexOld, int indexEnd, ACCESSOR2 hoodNew, int *indexNew, unsigned nanoStep)
    {
        updateLineXFluid(hoodOld, indexOld, indexEnd, hoodNew, indexNew);
    }

    template<typename ACCESSOR1, typename ACCESSOR2>
    static void updateLineXFluid(
        ACCESSOR1 hoodOld, int *indexOld, int indexEnd, ACCESSOR2 hoodNew, int *indexNew)
    {
#define GET_COMP(X, Y, Z, COMP) Double(hoodOld[FixedCoord<X, Y, Z>()].COMP())
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

        for (; *indexOld < indexEnd; *indexOld += ArityHelper<Double>::VALUE) {
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

            *indexNew += ArityHelper<Double>::VALUE;
        }
    }

    const double& read() const
    {
        return C;
    }

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

LIBFLATARRAY_REGISTER_SOA(LBMSoACell<double>, ((double)(C))((double)(N))((double)(E))((double)(W))((double)(S))((double)(T))((double)(B))((double)(NW))((double)(SW))((double)(NE))((double)(SE))((double)(TW))((double)(BW))((double)(TE))((double)(BE))((double)(TN))((double)(BN))((double)(TS))((double)(BS))((double)(density))((double)(velocityX))((double)(velocityY))((double)(velocityZ))((LBMSoACell<double>::State)(state)))

LIBFLATARRAY_REGISTER_SOA(LBMSoACell<ShortVec4xSSE>, ((double)(C))((double)(N))((double)(E))((double)(W))((double)(S))((double)(T))((double)(B))((double)(NW))((double)(SW))((double)(NE))((double)(SE))((double)(TW))((double)(BW))((double)(TE))((double)(BE))((double)(TN))((double)(BN))((double)(TS))((double)(BS))((double)(density))((double)(velocityX))((double)(velocityY))((double)(velocityZ))((LBMSoACell<ShortVec4xSSE>::State)(state)))

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
            ret->set(*i, CELL(1.0));
        }
        ret->setEdge(CELL(0.0));
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

    double seconds = 0.0;
    {
        ScopedTimer t(&seconds);

        sim.run();
    }

    if (sim.getGrid()->get(Coord<3>(1, 1, 1)).read() == 4711) {
        std::cout << "this statement just serves to prevent the compiler from"
                  << "optimizing away the loops above\n";
    }

    double updates = 1.0 * repeats * (dim - Coord<3>::diagonal(2)).prod();
    double glups = 10e-9 * updates / seconds;

    return glups;
}

template<class CELL>
void benchmark(std::string name, std::vector<Coord<3> > sizes)
{
    std::cout << "Benchmarking " << name << "\n";

    std::stringstream fileName;
    fileName << "data." << name << ".txt";
    std::ofstream outfile(fileName.str().c_str());
    outfile << "# BENCHMARK_ID MAXTRIX_SIZE GLUPS\n";

    int maxI = sizes.size();

    for (int i = 0; i < maxI; ++i) {
        Coord<3> c = sizes[i];
        int dim = c.x();
        double glups = singleBenchmark<CELL>(c);
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
    std::vector<Coord<3> > sizesLBM;
    sizesLBM << Coord<3>(22, 22, 22)
             << Coord<3>(64, 64, 64)
             << Coord<3>(68, 68, 68)
             << Coord<3>(106, 106, 106)
             << Coord<3>(128, 128, 128)
             << Coord<3>(160, 160, 160);

    std::vector<Coord<3> > sizesJacobi;
    for (int i = 0; i < 21; ++i) {
        int dim = std::pow(2, 4 + 0.25 * i);
        if (dim % 2) {
            ++dim;
        }

        sizesJacobi << Coord<3>::diagonal(dim);
    }

    benchmark<JacobiCellSimple           >("JacobiCellSimple", sizesJacobi);
    benchmark<JacobiCellMagic            >("JacobiCellMagic", sizesJacobi);
    benchmark<JacobiCellStraightforward  >("JacobiCellStraightforward", sizesJacobi);
    benchmark<JacobiCellStraightforwardNT>("JacobiCellStraightforwardNT", sizesJacobi);
    benchmark<JacobiCellStreakUpdate     >("JacobiCellStreakUpdate", sizesJacobi);
    benchmark<LBMSoACell<double>            >("LBMSoACell<double>", sizesLBM);
    benchmark<LBMSoACell<ShortVec4xSSE>     >("LBMSoACell<ShortVec4xSSE>", sizesLBM);

    return 0;
}
