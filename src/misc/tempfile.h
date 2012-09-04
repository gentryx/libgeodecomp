#ifndef _libgeodecomp_misc_tempfile_h_
#define _libgeodecomp_misc_tempfile_h_

#include <string>

namespace LibGeoDecomp {

class TempFile
{
public:
    static std::string serial(const std::string& prefix);
    static std::string parallel(const std::string& prefix);
};

}

#endif
