#include "mech.h"

Mech::Mech(int armor_, int ammo_, int power_) :
    armor(armor_), ammo(ammo_), power(power_)
{}

Car
Mech::transformToCar() {
    return Car();
}
