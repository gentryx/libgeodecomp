#ifndef LIBGEODECOMP_MISC_TEMPFILE_H
#define LIBGEODECOMP_MISC_TEMPFILE_H

#include <string>

namespace LibGeoDecomp {

/**
 * Simple wrapper to handle temporary files (somewhat) safely.
 */
class TempFile
{
public:
    static std::string serial(const std::string& prefix);
    static std::string parallel(const std::string& prefix);
    static bool exists(const std::string& filename);
    static void unlink(const std::string& filename);
};

}

#endif
