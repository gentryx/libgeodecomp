#ifndef _pair_h_
#define _pair_h_

#include "coord.h"

template<typename A, typename B>
class CoordPair
{
public:
    friend class Typemaps;
    A a;
    B b;
};

class Dummy
{
    friend class Typemaps;
public:
    CoordPair<int, int> p1;
    CoordPair<int, double> p2;
    CoordPair<Coord<3>, Coord<2> > p3;
};

#endif
