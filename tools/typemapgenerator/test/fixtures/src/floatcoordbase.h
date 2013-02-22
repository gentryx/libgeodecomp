#ifndef _floatcoord_h_
#define _floatcoord_h_

template<int DIMENSIONS>  
class FloatCoord
{
    friend class Typemaps;
private:
    float vec[DIMENSIONS];
};

class FloatCoordTypemapsHelper
{
    friend class Typemaps;
public:
    FloatCoord<1> pA;
    FloatCoord<2> pB;
    FloatCoord<4> pC;
};

#endif
