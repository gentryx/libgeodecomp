#ifndef LIBGEODECOMP_MISC_STRINGVEC_H
#define LIBGEODECOMP_MISC_STRINGVEC_H

// Kill some warnings in system headers:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4548 4710 4711 4996 )
#endif

#include <string>
#include <vector>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

typedef std::vector<std::string> StringVec;

}

#endif
