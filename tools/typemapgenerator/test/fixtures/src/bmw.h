#ifndef _BMW_h_
#define _BMW_h_

#include "car.h"
#include "luxury.h"

class BMW : public Car, private Luxury
{
public:
    friend class Typemaps;

    float price;
    
private:
    int serial;
};

#endif
