#ifndef _coord_h_
#define _coord_h_

template<int DIMENSIONS>  
class Coord;

class Coord<2>
{
    friend class Typemaps;
public:
    int x, y;
};

class Coord<3>
{
    friend class Typemaps;
public:
    int x, y, z;
};

#endif
