#ifndef _tire_h_
#define _tire_h_

class Tire
{
    friend class Typemaps;
    friend class BoostSerialization;
    friend class HPXSerialization;

    double treadDepth;
};

#endif
