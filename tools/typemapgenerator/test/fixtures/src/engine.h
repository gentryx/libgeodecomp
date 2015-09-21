#ifndef _engine_h_
#define _engine_h_

class Engine
{
    friend class Typemaps;
    friend class BoostSerialization;
    friend class HPXSerialization;

    enum Fuel {DIESEL, GASOLINE, AUTOGAS};

    double capacity;
    Fuel fuel;
    double gearRatios[6];
};

#endif
