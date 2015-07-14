#ifndef _LABEL_h_
#define _LABEL_h_

#include <string>

class Label
{
public:
    friend class BoostSerialization;
    friend class HPXSerialization;

    std::string name;
};

#endif
