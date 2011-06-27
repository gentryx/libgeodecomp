#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_partitioningsimulator_commjob_h_
#define _libgeodecomp_parallelization_partitioningsimulator_commjob_h_

#include <libgeodecomp/mpilayer/mpilayer.h>
#include <libgeodecomp/misc/supervector.h>

namespace LibGeoDecomp {

class CommJob
{
public:
    Coord<2> baseCoord;
    MPILayer::MPIRegionPointer region;
    unsigned partner;

    CommJob(Coord<2> baseCoord_, MPILayer::MPIRegionPointer region_, unsigned partner_) : 
        baseCoord(baseCoord_), region(region_), partner(partner_) {};
};

typedef SuperVector<CommJob> CommJobs;

};

#endif
#endif
