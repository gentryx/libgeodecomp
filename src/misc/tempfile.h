#ifndef LIBGEODECOMP_MISC_TEMPFILE_H
#define LIBGEODECOMP_MISC_TEMPFILE_H

#include <string>

namespace LibGeoDecomp {

/**
 * Simple wrapper to generate (somewhat) safe filenames for temporary files
 */
class TempFile
{
public:
    static std::string serial(const std::string& prefix);
    static std::string parallel(const std::string& prefix);
};

}

#endif
