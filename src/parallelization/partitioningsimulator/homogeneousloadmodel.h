#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_partitioningsimulator_homogeneousloadmodel_h_
#define _libgeodecomp_parallelization_partitioningsimulator_homogeneousloadmodel_h_

#include <libgeodecomp/parallelization/partitioningsimulator/loadmodel.h>

namespace LibGeoDecomp {

class HomogeneousLoadModel : public LoadModel
{
    friend class HomogeneousLoadModelTest;

public:
    HomogeneousLoadModel(
            MPILayer* mpilayer, 
            const unsigned& master,
            const unsigned& historySize = 32); 

    ~HomogeneousLoadModel();

    DVec powers(const unsigned& step = 0) const;

    double weight(const CoordBox<2>& rect) const;

    std::string report() const;

    void sync(const Partition& partition, const double& time);

protected:
    SuperVector<DVec> _powerHistory;

};

};

#endif
#endif
