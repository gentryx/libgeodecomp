#ifndef _car_h_
#define _car_h_

#include "engine.h"
#include "mech.h"
#include "wheel.h"

class Mech;

class Car
{
    friend class Typemaps;

    static const unsigned NumWheels = 4;
    Wheel wheels[NumWheels];
    Engine engine;

public:
    Mech transformToMech();
};

#endif
