#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_partitioningsimulator_statebasedloadmodel_h_
#define _libgeodecomp_parallelization_partitioningsimulator_statebasedloadmodel_h_

#include <libgeodecomp/misc/displacedgrid.h>
#include <libgeodecomp/parallelization/partitioningsimulator/commjob.h>
#include <libgeodecomp/parallelization/partitioningsimulator/homogeneousloadmodel.h>
#include <libgeodecomp/parallelization/partitioningsimulator/nnlsfit.h>

namespace LibGeoDecomp {

/**
 * Keeps the concept of varying powers from HomogeneousLoadModel but
 * abandons the assumption of homogeneous costs
 */
class StateBasedLoadModel : public HomogeneousLoadModel
{
    friend class StateBasedLoadModelTest;
    friend class ModelSplitterTest;

public:
    StateBasedLoadModel(
            MPILayer* mpilayer, 
            const unsigned& master,
            const CoordBox<2>& global,
            const unsigned& numStates,
            const unsigned& historySize = 32); 

    ~StateBasedLoadModel();

    double weight(const CoordBox<2>& rect) const;

    /**
     * must be called whenever the host node is repartitioned before cell
     * states are registered
     */
    void repartition(const Partition& newPartition);

    inline void registerCellState(const Coord<2>& coord, const StateType& state) 
    {
        if (!_localGrid->boundingBox().inBounds(coord)) {
            throw std::invalid_argument(
                    "StateBasedLoadModel::registerCellState: coord " 
                    + coord.toString() + " out of bounds for local grid " 
                    + _localGrid->boundingBox().toString() 
                    + ". Did you forget to call repartition?");
        }

        (*_localGrid)[coord] = state;
        _stateCounts[state]++;
    }

    void sync(const Partition& partition, const double& time);

    std::string report() const;
    std::string summary();

private:
    unsigned _numStates;

    DisplacedGrid<StateType>* _localGrid;
    DisplacedGrid<StateType>* _globalGrid;

    SuperVector<DVec> _timingHistory;
    DVec _stateCounts;
    DVec _averageCosts;

    NNLSFit* _fitCosts;

    // fixme: mutable?  why?
    mutable Chronometer _totalGatherGlobalGridTimer;
    mutable Chronometer _totalCollectStateCountsTimer;

    CommJobs _sendJobs;
    CommJobs _recvJobs;
    SuperVector<MPILayer::MPIRegionPointer> _registeredRegions;

    static DVec norm(const DVec& v);

    static double scalarProduct(const DVec& v0, const DVec& v1);

    void gatherGlobalGrid();

    DVec estimateLocalCost(const double& time);

    DVec gatherAverageCosts(const DVec& localCost);

    DVec collectStateCounts(const CoordBox<2>& rect) const;
};

};

#endif
#endif
