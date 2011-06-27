#ifndef _BMW_h_
#define _BMW_h_

#include "car.h"
#include "luxury.h"

class BMW : public Car, private Luxury
{
    friend class Typemaps;
public:
    float price;
private:
    int serial;
};

#endif
