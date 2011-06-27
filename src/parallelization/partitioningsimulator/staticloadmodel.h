#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_partitioningsimulator_staticloadmodel_h_
#define _libgeodecomp_parallelization_partitioningsimulator_staticloadmodel_h_

#include <libgeodecomp/parallelization/partitioningsimulator/loadmodel.h>

namespace LibGeoDecomp {

class StaticLoadModel : public LoadModel
{

public:
    StaticLoadModel(MPILayer* mpilayer, const unsigned& master):
        LoadModel(mpilayer, master)
    {}

    ~StaticLoadModel() {}

    DVec powers(const unsigned&) const 
    { 
        return DVec(_mpilayer->size(), 1); 
    }


    double weight(const CoordBox<2>& rect) const
    {
        return rect.size();
    }


    double stepWeight(const unsigned&) const
    {
        return 1.0;
    }


    double predictRunningTime(const Partition&, const unsigned&) const
    {
        return 0;
    }
};

};

#endif
#endif
