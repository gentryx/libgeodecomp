#ifndef _echo_h_
#define _echo_h_

#include <string>
#include <iostream>

class Echo
{
    friend class Typemaps;
public:
    void echo(std::string s);
    void ping() { std::cout <<  "pong\n"; }

    int i;
};

#endif
