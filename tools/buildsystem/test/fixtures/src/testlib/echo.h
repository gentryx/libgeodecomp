#ifndef _echo_h_
#define _echo_h_

#include <string>
#include <iostream>

class Echo
{
public:
    void echo(std::string s);
    void ping() { std::cout <<  "ping\n"; }
};

#endif
