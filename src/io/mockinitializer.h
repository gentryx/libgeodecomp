#ifndef LIBGEODECOMP_IO_MOCKINITIALIZER_H
#define LIBGEODECOMP_IO_MOCKINITIALIZER_H

#include <libgeodecomp/io/testinitializer.h>

namespace LibGeoDecomp {

class MockInitializer : public TestInitializer<TestCell<2> >
{
public:
    MockInitializer(const std::string& configString = "") 
    { 
        events += "created, configString: '" + configString + "'\n";
    }

    ~MockInitializer() 
    { 
        events += "deleted\n"; 
    }

    static std::string events;
};

}

#endif
