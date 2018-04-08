#ifndef LIBGEODECOMP_MISC_TEMPFILE_H
#define LIBGEODECOMP_MISC_TEMPFILE_H

// Kill warning 4514 in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <string>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

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
