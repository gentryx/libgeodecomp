#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_partitioningsimulator_loadmodel_h_
#define _libgeodecomp_parallelization_partitioningsimulator_loadmodel_h_

#include <stdexcept>
#include <libgeodecomp/misc/commontypedefs.h>
#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/mpilayer/mpilayer.h>
#include <libgeodecomp/parallelization/chronometer.h>
#include <libgeodecomp/parallelization/partitioningsimulator/partition.h>

namespace LibGeoDecomp {

/**
 * Base Class for generating observations for a node and forming a combined model based on
 * the observations of all nodes
 */
class LoadModel
{
    friend class HomogeneousLoadModelTest;

public:
    typedef unsigned char StateType;

    LoadModel(MPILayer* mpilayer, const unsigned& master);

    virtual ~LoadModel();

    /**
     * returns powers estimate for @a step. step 0 is the most recent step, step
     * 1 the previous step and so forth...
     */
    virtual DVec powers(const unsigned& step = 0) const = 0;

    virtual double weight(const CoordBox<2>& rect) const = 0;

    /**
     * Predicts running time over the next @a timeSteps by using the tried and
     * true axiom of conservativity: "The Future Will be like the Past".
     */
    double predictRunningTime(const Partition& partition) const;


    double expectedGainFromPartitioning(
            const Partition& oldPartition, 
            const Partition& newPartition,
            const unsigned& maxHorizon = 32) const;

    /**
     * must be called whenever the host node is repartitioned before cell
     * states are registered
     */
    virtual void repartition(const Partition&);

    virtual void registerCellState(const Coord<2>&, const StateType&);

    virtual void sync(const Partition&, const double&);

    /**
     * report on the current status
     */
    virtual std::string report() const;

    /*
     * call this to get a final summary of LoadModel behavior
     */
    virtual std::string summary();

protected:
    MPILayer* _mpilayer;
    unsigned _master;

    void verifyMaster() const;
    inline unsigned numNodes() const { return _mpilayer->size(); }

    std::string timerSummary(
            const std::string& description, Chronometer& timer);

private:
    double expectedRunningTime(
        const DVec& nodeWeights, const unsigned& step) const;
};

};

#endif
#endif
