#ifndef _LABEL_h_
#define _LABEL_h_

#include <string>

class Label
{
    friend class Serialize;
public:
    std::string name;
};

#endif
