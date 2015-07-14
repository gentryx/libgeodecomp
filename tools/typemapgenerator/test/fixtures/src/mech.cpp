#include "mech.h"

Mech::Mech(int armor, int ammo, int power) :
    armor(armor),
    ammo(ammo),
    power(power)
{}

Car
Mech::transformToCar() {
    return Car();
}
