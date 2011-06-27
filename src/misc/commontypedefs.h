#ifndef _libgeodecomp_misc_commontypedefs_h_
#define _libgeodecomp_misc_commontypedefs_h_

#include <libgeodecomp/misc/supervector.h>
#include <libgeodecomp/misc/superset.h>

namespace LibGeoDecomp {
typedef SuperVector<unsigned> UVec;
typedef SuperVector<int> IVec;
typedef SuperVector<double> DVec;
typedef SuperVector<long long> LLVec;

typedef SuperSet<unsigned> USet;

typedef std::pair<unsigned, unsigned> UPair;
typedef std::pair<double, double> DPair;
};

#endif
