#ifndef LIBGEODECOMP_TESTBED_INTERFACES
#define LIBGEODECOMP_TESTBED_INTERFACES

#include <emmintrin.h>
#include <libgeodecomp/misc/grid.h>

using namespace LibGeoDecomp;

template<int DIM_X=0, int DIM_Y=0, int DIM_Z=0>
class FixedCoord
{
public:
};

typedef Grid<double, Topologies::Cube<2>::Topology> GridType;

class HoodOld;

class CellInterface
{
public:
    virtual void update(HoodOld& hood) = 0;
    virtual double getVal() = 0;
};

class CellVirtual : public CellInterface
{
public:   
    CellVirtual(double v=0);
    virtual void update(HoodOld&);
    virtual double getVal();

private:
    double val;
};

typedef Grid<CellVirtual, Topologies::Cube<2>::Topology> VirtualGridType;

class HoodOld
{
public:
    HoodOld(VirtualGridType *_grid, Coord<2> _center) :
        grid(_grid),
        center(_center)
    {}

    CellInterface& operator[](Coord<2> &c)
    {
        return (*grid)[c + center];
    }

private:
    VirtualGridType *grid;
    Coord<2> center;
};

class CellStraight
{
public:   
    CellStraight(double v=0) : 
        val(v)
    {}

    template<typename HOOD>
    void update(const HOOD& hood)
    {
        val = 
            (hood[Coord<2>( 0, -1)].val +
             hood[Coord<2>(-1,  0)].val +
             hood[Coord<2>( 0,  0)].val +
             hood[Coord<2>( 1,  0)].val +
             hood[Coord<2>( 0,  1)].val) * (1.0 / 5.0);
    }

    template<typename HOOD>
    void update2(const HOOD& hood)
    {
        val = 
            (hood[FixedCoord< 0, -1>()].val +
             hood[FixedCoord<-1,  0>()].val +
             hood[FixedCoord< 0,  0>()].val +
             hood[FixedCoord< 1,  0>()].val +
             hood[FixedCoord< 0,  1>()].val) * (1.0 / 5.0);
    }

    template<typename HOOD>
    static void updateStreak(CellStraight *c, const HOOD& hood, const int& startX, const int& endX)
    {
        // for (int x = startX; x < endX; ++x) {
        //     c[x].val = 
        //         (hood[FixedCoord<0, -1>()][x + 0].val +
        //          hood[FixedCoord<0,  0>()][x - 1].val +
        //          hood[FixedCoord<0,  0>()][x + 0].val +
        //          hood[FixedCoord<0,  0>()][x + 1].val +
        //          hood[FixedCoord<0,  1>()][x + 0].val) * (1.0 / 5.0);
        // }

#define hoody(X, Y) hood[FixedCoord<X, Y>()][x]

        for (int x = startX; x < endX; ++x) {
            c[x].val = 
                (hoody( 0, -1).val +
                 hoody(-1,  0).val +
                 hoody( 0,  0).val +
                 hoody( 1,  0).val +
                 hoody( 0,  1).val) * (1.0 / 5.0);
        }
    }

    template<typename HOOD>
    static void updateStreakSSE(CellStraight *c, const HOOD& hood, const int& startX, const int& endX)
    {
        int x = startX;

        if (x % 2) {
            updateStreak(c, hood, x, x + 1);
            x += 1;
        }

        __m128d buf0 = _mm_load_pd( &hoody(-2, 0).val);
        __m128d buf1 = _mm_load_pd( &hoody( 0, 0).val);
        __m128d tmp0 = _mm_loadu_pd(&hoody(-1, 0).val);

        __m128d oneFifth = _mm_set_pd(1.0/5.0, 1.0/5.0);

        for (; x < endX - 7; x += 8) {
            __m128d buf2 = _mm_load_pd(&hoody(2, 0).val);
            __m128d buf3 = _mm_load_pd(&hoody(4, 0).val);
            __m128d buf4 = _mm_load_pd(&hoody(6, 0).val);
            __m128d buf5 = _mm_load_pd(&hoody(8, 0).val);
                      
            __m128d xxx = buf4;
                 
            __m128d tmp1 = _mm_shuffle_pd(buf1, buf2, (1 << 0) | (0 << 2));
            __m128d tmp2 = _mm_shuffle_pd(buf2, buf3, (1 << 0) | (0 << 2));
            __m128d tmp3 = _mm_shuffle_pd(buf3, buf4, (1 << 0) | (0 << 2));
            __m128d tmp4 = _mm_shuffle_pd(buf4, buf5, (1 << 0) | (0 << 2));
            
            __m128d foo1 = _mm_load_pd(&hoody(0, -1).val);
            __m128d foo2 = _mm_load_pd(&hoody(2, -1).val);
            __m128d foo3 = _mm_load_pd(&hoody(4, -1).val);
            __m128d foo4 = _mm_load_pd(&hoody(6, -1).val);

            buf1 = _mm_add_pd(buf1, tmp0);
            buf2 = _mm_add_pd(buf2, tmp1);
            buf3 = _mm_add_pd(buf3, tmp2);
            buf4 = _mm_add_pd(buf4, tmp3);

            buf1 = _mm_add_pd(buf1, foo1);
            buf2 = _mm_add_pd(buf2, foo2);
            buf3 = _mm_add_pd(buf3, foo3);
            buf4 = _mm_add_pd(buf4, foo4);

            foo1 = _mm_load_pd(&hoody(0, 1).val);
            foo2 = _mm_load_pd(&hoody(2, 1).val);
            foo3 = _mm_load_pd(&hoody(4, 1).val);
            foo4 = _mm_load_pd(&hoody(6, 1).val);

            buf1 = _mm_add_pd(buf1, tmp1);
            buf2 = _mm_add_pd(buf2, tmp2);
            buf3 = _mm_add_pd(buf3, tmp3);
            buf4 = _mm_add_pd(buf4, tmp4);

            buf1 = _mm_add_pd(buf1, foo1);
            buf2 = _mm_add_pd(buf2, foo2);
            buf3 = _mm_add_pd(buf3, foo3);
            buf4 = _mm_add_pd(buf4, foo4);

            buf1 = _mm_mul_pd(buf1, oneFifth);
            buf2 = _mm_mul_pd(buf2, oneFifth);
            buf3 = _mm_mul_pd(buf3, oneFifth);
            buf4 = _mm_mul_pd(buf4, oneFifth);

            _mm_store_pd(&c[x + 0].val, buf1);
            _mm_store_pd(&c[x + 2].val, buf2);
            _mm_store_pd(&c[x + 4].val, buf3);
            _mm_store_pd(&c[x + 6].val, buf4);

            tmp0 = tmp4;
            buf0 = xxx;
            buf1 = buf5;
        }

        updateStreak(c, hood, x, endX);
    }

    double val;
};

typedef Grid<CellStraight, Topologies::Cube<2>::Topology> StraightGridType;

class HoodStraight
{
public:
    HoodStraight(StraightGridType *_grid, Coord<2> _center) :
        grid(_grid),
        center(_center)
    {}

    const CellStraight& operator[](const Coord<2> &c) const
    {
        return (*grid)[c + center];
    }

private:
    StraightGridType *grid;
    Coord<2> center;
};

class HoodSteak
{
public:
    HoodSteak(CellStraight **_lines, int *_offset) :
        lines(_lines),
        offset(_offset)
    {}

    template<int X, int Y>
    inline CellStraight& operator[](const FixedCoord<X, Y, 0>&) const
    {
        return lines[1 + Y][X + *offset];
    }

private:
    CellStraight **lines;
    int *offset;

};

class HoodStreak
{
public:
    HoodStreak(CellStraight **_lines) :
        lines(_lines)
    {}

    template<int X, int Y>
    inline CellStraight *operator[](const FixedCoord<X, Y, 0>&) const
    {
        return lines[1 + Y] + X;
    }

private:
    CellStraight **lines;

};

#endif
