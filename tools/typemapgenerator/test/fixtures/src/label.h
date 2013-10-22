#ifndef _LABEL_h_
#define _LABEL_h_

#include <string>

class Label
{
    friend class Serialization;
public:
    std::string name;
};

#endif
