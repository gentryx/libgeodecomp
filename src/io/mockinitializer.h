#ifndef _libgeodecomp_io_mockinitializer_h_
#define _libgeodecomp_io_mockinitializer_h_

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
