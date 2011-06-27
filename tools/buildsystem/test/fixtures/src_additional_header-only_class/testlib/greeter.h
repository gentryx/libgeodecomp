#ifndef _greeter_h_
#define _greeter_h_

#include "iostream"

class Greeter
{
    friend class Typemaps;
public:
    void hello(std::string name = "andi") {
        std::cout << "hello " << name << "\n";
    }

    int deathToll;
};

#endif
