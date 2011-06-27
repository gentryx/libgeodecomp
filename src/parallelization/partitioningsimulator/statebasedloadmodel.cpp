#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#include <libgeodecomp/parallelization/partitioningsimulator/statebasedloadmodel.h>

namespace LibGeoDecomp {

StateBasedLoadModel::StateBasedLoadModel(
            MPILayer* mpilayer, 
            const unsigned& master,
            const CoordBox<2>& global,
            const unsigned& numStates,
            const unsigned& historySize):
    HomogeneousLoadModel(mpilayer, master, historySize),
    _numStates(numStates),
    _localGrid(0),
    _globalGrid(0),
    _timingHistory(),
    _stateCounts(numStates),
    _averageCosts(numStates, 0),
    _fitCosts(0)
{
    // repartition must be called before registering cellStates
    _localGrid = new DisplacedGrid<StateType>();
    if (_mpilayer->rank() == _master) 
        _globalGrid = new DisplacedGrid<StateType>(global);

    unsigned entrySize = _numStates + 1;
    for (unsigned i = 0; i < _numStates * 2; i++) 
        _timingHistory.push_back(DVec(entrySize, 0));

    _fitCosts = new NNLSFit(_timingHistory);
}


StateBasedLoadModel::~StateBasedLoadModel()
{
    delete _fitCosts;
    delete _localGrid;
    if (_globalGrid) delete _globalGrid;
} 


double StateBasedLoadModel::weight(const CoordBox<2>& rect) const
{
    verifyMaster();
    DVec stateCounts = collectStateCounts(rect);
    return scalarProduct(_averageCosts, stateCounts);
}


std::string StateBasedLoadModel::report() const
{
    verifyMaster();

    /*
     *DVec globalCounts(_numStates);
     *CoordBox<2> rect = _globalGrid->boundingBox();
     *for (CoordBox<2>Sequence s = rect.sequence(); s.hasNext();)
     *    globalCounts[(*_globalGrid)[s.next()]]++;
     */

    std::ostringstream msg;
    msg << "powers: " << powers() << "\n"
        << "average costs: " << _averageCosts << "\n";
        //<< "global state counts: " << globalCounts << "\n";
    return msg.str();
}


void StateBasedLoadModel::sync(const Partition& partition, const double& time)
{
    // gather data
    _averageCosts = gatherAverageCosts(estimateLocalCost(time));
    gatherGlobalGrid();
    DVec times = _mpilayer->gather(time, _master);

    // reset counter
    _stateCounts.assign(_stateCounts.size(), 0);

    // process data
    if (_mpilayer->rank() == _master) {
        DVec newPowers(powers());
        for (unsigned i = 0; i < numNodes(); i++) {
            double time = times[i];
            double nodeWeight = weight(partition.coordsForNode(i));
            if (time > 0 && nodeWeight > 0) newPowers[i] = nodeWeight / time;
        }
        _powerHistory.erase(_powerHistory.begin());
        _powerHistory.push_back(newPowers);
    }
}
 

void StateBasedLoadModel::repartition(const Partition& newPartition)
{
    // _globalGrid never changes    
    CoordBox<2> local = newPartition.coordsForNode(_mpilayer->rank());
    if (_localGrid->boundingBox() != local) {
        delete _localGrid;
        _localGrid = new DisplacedGrid<StateType>(local);
    }

    _sendJobs.clear();
    _recvJobs.clear();
    Nodes nodes = newPartition.getNodes();
    unsigned rank = _mpilayer->rank();


    CoordBox<2> localRect = newPartition.coordsForNode(rank);
    localRect.origin -= _localGrid->boundingBox().origin;
    Coord<2> base = localRect.origin;
    MPILayer::MPIRegionPointer rp = _mpilayer->registerRegion(
        *_localGrid->vanillaGrid(), localRect, base);
    _sendJobs.push_back(CommJob(base, rp, _master));

    if (rank == _master) {
        Coord<2> base = _globalGrid->boundingBox().origin;
        for (Nodes::iterator i = nodes.begin(); i != nodes.end(); i++) {
            CoordBox<2> localRect = newPartition.coordsForNode(*i);
            localRect.origin -= _globalGrid->boundingBox().origin;
            MPILayer::MPIRegionPointer rp = _mpilayer->registerRegion(
                    *_globalGrid->vanillaGrid(), localRect, base);
            _recvJobs.push_back(CommJob(base, rp, *i));
        }
    }
}


void StateBasedLoadModel::gatherGlobalGrid()
{
    _totalGatherGlobalGridTimer.tic();

    for(CommJobs::const_iterator i = _sendJobs.begin(); 
            i != _sendJobs.end(); i++) {
        _mpilayer->sendRegion(&((*_localGrid)[i->baseCoord]),
                i->region, i->partner);
    }

    for(CommJobs::const_iterator i = _recvJobs.begin(); 
            i != _recvJobs.end(); i++) {
        _mpilayer->recvRegion(&((*_globalGrid)[i->baseCoord]),
                i->region, i->partner);
    }
    _mpilayer->waitAll();

    _totalGatherGlobalGridTimer.toc();
}


DVec StateBasedLoadModel::estimateLocalCost(const double& time)
{
    DVec latestTiming(_stateCounts);
    latestTiming.push_back(time);
    _timingHistory.erase(_timingHistory.begin());
    _timingHistory.push_back(latestTiming);
    _fitCosts->refit(_timingHistory);
    DVec coefficients = _fitCosts->coefficients();
    return norm(coefficients);
}


DVec StateBasedLoadModel::gatherAverageCosts(const DVec& localCost) 
{
    DVec averageCosts(_numStates);
    for (unsigned i = 0; i < _numStates; i++)
    {
        DVec costs = _mpilayer->gather(localCost[i], _master);
        averageCosts[i] = costs.sum() / costs.size();
    }
    return averageCosts;
}


DVec StateBasedLoadModel::norm(const DVec& v)
{
    DVec result(v.size());
    double length = 0;
    for (unsigned i = 0; i < v.size(); i++) 
        length += v[i] * v[i];
    length = std::sqrt(length);
    for (unsigned i = 0; i < v.size(); i++) 
        result[i] = v[i] / length;
    return result;
}


double StateBasedLoadModel::scalarProduct(const DVec& v0, const DVec& v1)
{
    if (v0.size() != v1.size())
        throw std::invalid_argument(
                "LoadModel::scalarProduct: sizes don't match");
    double result = 0;
    for (unsigned i = 0; i < v0.size(); i++)
        result += v0[i] * v1[i];
    return result;
}


DVec StateBasedLoadModel::collectStateCounts(const CoordBox<2>& rect) const
{
    _totalCollectStateCountsTimer.tic();
    verifyMaster();
    DVec counts(_numStates);
    for (CoordBoxSequence<2> s = rect.sequence(); s.hasNext();)
        counts[(*_globalGrid)[s.next()]]++;
    _totalCollectStateCountsTimer.toc();
    return counts;
}


std::string StateBasedLoadModel::summary()
{
    std::ostringstream tmp;
    tmp << timerSummary("gathering global grid", _totalGatherGlobalGridTimer)
        << timerSummary("collecting state counts", _totalCollectStateCountsTimer);
    return tmp.str();
}

};
#endif
