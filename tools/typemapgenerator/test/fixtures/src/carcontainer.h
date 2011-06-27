#ifndef _carcontainer_h_
#define _carcontainer_h_

#include "car.h"

/**
 * The CarContainer class is meant to demonstrate the
 * TypemapGenerator's ability to create partial MPI datatypes for
 * classes containing pointer members.
 */
class CarContainer {
    friend class Typemaps;

    Car *alpha;
    Car *bravo;
    Car *charly;
    Car *delta;
    Car *echo;

    int size;
    Wheel spareWheel;
};

#endif
