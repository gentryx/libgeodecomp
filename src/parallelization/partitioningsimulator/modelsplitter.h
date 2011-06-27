#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_partitioningsimulator_modelsplitter_h_
#define _libgeodecomp_parallelization_partitioningsimulator_modelsplitter_h_

#include <libgeodecomp/parallelization/partitioningsimulator/splitter.h>
#include <libgeodecomp/parallelization/partitioningsimulator/loadmodel.h>

namespace LibGeoDecomp {

/**
 * This splitter uses a LoadModel to calculate its splits
 */
class ModelSplitter : public Splitter
{
    friend class ModelSplitterTest;

public:
    ModelSplitter(
        const LoadModel* model, 
        const SplitDirection& direction = LONGEST):
        Splitter(model->powers(), direction),
        _loadModel(model)
    {}


    ModelSplitter(
        const LoadModel* model, 
        const ClusterTable& table,
        const SplitDirection& direction = LONGEST):
        Splitter(model->powers(), table, direction),
        _loadModel(model)
    {}

protected:

    virtual DVec powers() const 
    { 
        return _loadModel->powers(); 
    }


    virtual double weight(const CoordBox<2>& rect) const
    {
        return _loadModel->weight(rect);
    }


private:
    const LoadModel* _loadModel;
};

};

#endif
#endif
