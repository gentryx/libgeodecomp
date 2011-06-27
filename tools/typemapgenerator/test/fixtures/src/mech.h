#ifndef _mech_h_
#define _mech_h_

#include "car.h"

class Car;

class Mech
{
    int armor;
    int ammo;
    int power;

public:
    Mech(int armor_, int ammo_, int power_);
    Car transformToCar();
};

#endif
