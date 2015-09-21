#ifndef _mech_h_
#define _mech_h_

#include "car.h"

class Car;

class Mech
{
public:
    Mech(int armor, int ammo, int power);

    Car transformToCar();

private:
    int armor;
    int ammo;
    int power;
};

#endif
