#ifndef _wheel_h_
#define _wheel_h_

#include "tire.h"
#include "rim.h"

class Wheel 
{
    friend class Typemaps;
    Tire tire;
    Rim rim;
};

#endif
