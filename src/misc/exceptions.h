#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_misc_exceptions_h_
#define _libgeodecomp_misc_exceptions_h_

#include <stdexcept>
#include <string>

namespace LibGeoDecomp {

/**
 * Custom exception class which is suitable for all usage errors
 */
class UsageException : public std::runtime_error 
{
    public: 
    UsageException(const std::string& what) : std::runtime_error(what) {}
};

};

#endif
#endif
