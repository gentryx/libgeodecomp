#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#include <math.h>
#include <libgeodecomp/parallelization/partitioningsimulator/communicationmodel.h>

namespace LibGeoDecomp {

CommunicationModel::CommunicationModel(const ClusterTable& table):
    _fit(0),
    _numObservations(6), // slightly overdetermined
    _observations(),
    _table(table)
{
/*
 *    // costs are initialized to 0
 *    for (unsigned i = 0; i < _numObservations; i++) {
 *        DVec dataPoint;
 *        dataPoint.push_back(1); // constant term
 *        dataPoint.push_back(i); // maxCommunication
 *        dataPoint.push_back(0); // interClusterComm
 *        dataPoint.push_back(0); // time
 *        _observations.push_back(dataPoint);
 *    }
 *
 */
    _fit = new NNLSFit(NNLSFit::DataPoints(_numObservations, DVec(4)));
}


CommunicationModel::~CommunicationModel()
{
    delete _fit;
}


double CommunicationModel::predictRepartitionTime(
        const Partition& pOld, const Partition& pNew)
{
    throwUnlessSizesMatch(pOld, pNew);
    if (_observations.size() == _numObservations) {
        return _fit->solve(assembleParams(pOld, pNew));
    } else {
        return 0; // make sure we gather enough observations
    }
}


void CommunicationModel::addObservation(
        const Partition& pOld, const Partition& pNew, const DVec& workLengths)
{
    throwUnlessSizesMatch(pOld, pNew);

    DVec dataPoint = assembleParams(pOld, pNew);
    dataPoint.push_back(workLengths.max());
    _observations.push_back(dataPoint);

    if (_observations.size() > _numObservations) 
        _observations.erase(_observations.begin());
    
    if (_observations.size() == _numObservations) 
        _fit->refit(_observations); // only refit if we have enough observations
}


DVec CommunicationModel::assembleParams(
        const Partition& p0, const Partition& p1) const
{
    DVec params;
    params.push_back(1); // constant term
    params.push_back(maxCommunication(p0, p1));
    params.push_back(interClusterComm(p0, p1, _table));
    return params;
}


unsigned CommunicationModel::intersectedArea(
        const Partition& p0, const Partition& p1) 
{
    Nodes nodes = p0.getNodes();
    unsigned result = 0;
    for (Nodes::iterator i = nodes.begin(); i != nodes.end(); i++) 
        result += p0.coordsForNode(*i).intersect(p1.coordsForNode(*i)).size();

    return result;
}


unsigned CommunicationModel::maxCommunication(
        const Partition& p0, const Partition& p1)
{
    Nodes nodes = p0.getNodes();
    unsigned result = 0;
    for (Nodes::iterator i = nodes.begin(); i != nodes.end(); i++) {
        CoordBox<2> rect0 = p0.coordsForNode(*i);
        CoordBox<2> rect1 = p1.coordsForNode(*i);
        unsigned intersection = rect0.intersect(rect1).size();
        unsigned symDifference = rect0.size() + rect1.size() - 2 * intersection;
        result = std::max(result, symDifference);
    }
    return result;
}


unsigned CommunicationModel::interClusterComm(
            const Partition& p0, const Partition& p1, const ClusterTable& table)
{
    Nodes nodes = p0.getNodes();
    unsigned result = 0;
    for (ClusterTable::const_iterator i = table.begin(); 
            i != table.end(); i++) {
        CoordBox<2> rect0 = p0.rectForNodes(*i);
        CoordBox<2> rect1 = p1.rectForNodes(*i);
        result += rect0.size() - rect0.intersect(rect1).size();
    }
    return result;
}


void CommunicationModel::throwUnlessSizesMatch(
        const Partition& p0, const Partition& p1) 
{
    if (p0.getRect() != p1.getRect())
        throw std::invalid_argument("CommunicationModel: Partition sizes do not match");
    if (p0.getNodes() != p1.getNodes())
        throw std::invalid_argument("CommunicationModel: Nodes do not match");
}


std::string CommunicationModel::report() const
{
    std::ostringstream msg;
    msg << "CommunicationModel coefficients: " << _fit->coefficients() << "\n";
    return msg.str();
}


std::string CommunicationModel::summary()
{
    std::ostringstream tmp;
    return tmp.str();
}

};
#endif
