#ifndef _wheel_h_
#define _wheel_h_

#include "tire.h"
#include "rim.h"

class Wheel
{
    friend class Typemaps;
    friend class BoostSerialization;
    friend class HPXSerialization;
    Tire tire;
    Rim rim;
};

#endif
