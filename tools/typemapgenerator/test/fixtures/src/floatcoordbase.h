#ifndef _floatcoord_h_
#define _floatcoord_h_

template<int DIMENSIONS>
class FloatCoord
{
private:
    friend class Typemaps;

    float vec[DIMENSIONS];
};

class FloatCoordTypemapsHelper
{
public:
    friend class Typemaps;

    FloatCoord<1> pA;
    FloatCoord<2> pB;
    FloatCoord<4> pC;
};

#endif
