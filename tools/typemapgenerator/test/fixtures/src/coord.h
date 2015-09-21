#ifndef _coord_h_
#define _coord_h_

template<int DIMENSIONS>
class Coord;

class Coord<1>
{
public:
    friend class BoostSerialization;
    friend class HPXSerialization;

    int x;
};

class Coord<2>
{
public:
    friend class Typemaps;

    int x, y;
};

class Coord<3>
{
public:
    friend class Typemaps;
    friend class BoostSerialization;
    friend class HPXSerialization;

    int x, y, z;
};

#endif
