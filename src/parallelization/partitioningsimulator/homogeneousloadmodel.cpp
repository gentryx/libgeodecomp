#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#include <libgeodecomp/parallelization/partitioningsimulator/homogeneousloadmodel.h>

namespace LibGeoDecomp {

HomogeneousLoadModel::HomogeneousLoadModel(
            MPILayer* mpilayer, 
            const unsigned& master,
            const unsigned& historySize):
    LoadModel(mpilayer, master),
    _powerHistory(historySize, DVec(mpilayer->size(), 1.0 / mpilayer->size()))
{}


HomogeneousLoadModel::~HomogeneousLoadModel()
{} 


DVec HomogeneousLoadModel::powers(const unsigned& step) const
{
    if (step >= _powerHistory.size())
        return DVec(_mpilayer->size(), 1.0 / _mpilayer->size());

    verifyMaster();
    return _powerHistory.at(_powerHistory.size() - 1 - step); 
}


double HomogeneousLoadModel::weight(const CoordBox<2>& rect) const
{
    return rect.size();
}


std::string HomogeneousLoadModel::report() const
{
    verifyMaster();
    std::ostringstream msg;
    msg << "powers: " << powers() << "\n";
    return msg.str();
}


void HomogeneousLoadModel::sync(const Partition& partition, const double& time)
{
    DVec times = _mpilayer->gather(time, _master);

    if (_mpilayer->rank() == _master) {

        DVec newPowers(powers());
        for (unsigned i = 0; i < numNodes(); i++) {
            double time = times[i];
            unsigned size = partition.coordsForNode(i).size();
            if (time > 0 && size > 0) newPowers[i] = size / time;
        }

        _powerHistory.erase(_powerHistory.begin());
        _powerHistory.push_back(newPowers);
    }
}

};
#endif
