#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#include <libgeodecomp/misc/stringops.h> 
#include <libgeodecomp/parallelization/partitioningsimulator/loadmodel.h>

namespace LibGeoDecomp {

LoadModel::LoadModel(MPILayer* mpilayer, const unsigned& master): 
    _mpilayer(mpilayer), _master(master)
{
    unsigned size = _mpilayer->size(); 
    if (_master >= size) {
        // fixme: use boost::lexical_cast here
        throw std::invalid_argument("master rank " + StringConv::itoa(_master)
                                    + " out of range for mpilayer.size() " + StringConv::itoa(size));
    }
}

LoadModel::~LoadModel() 
{}


void LoadModel::repartition(const Partition&)
{}


void LoadModel::sync(const Partition&, const double&) 
{}


void LoadModel::registerCellState(const Coord<2>&, const StateType&)
{}


std::string LoadModel::report() const
{ 
    return ""; 
}


std::string LoadModel::summary() 
{ 
    return ""; 
}

/**
 * PLBa
 */
/*
 *double LoadModel::expectedGainFromPartitioning(
 *        const Partition& oldPartition, 
 *        const Partition& newPartition,
 *        const unsigned& maxHorizon) const
 *{
 *    verifyMaster();
 *    if (!oldPartition.compatible(newPartition)) 
 *        throw std::invalid_argument("Partitions incompatible");
 *    
 *    Nodes nodes = oldPartition.getNodes();
 *    DVec oldWeights;
 *    DVec newWeights;
 *    oldWeights.reserve(nodes.size());
 *    newWeights.reserve(nodes.size());
 *    for (Nodes::iterator i = nodes.begin(); i != nodes.end(); i++) {
 *        oldWeights.push_back(weight(oldPartition.coordsForNode(*i)));
 *        newWeights.push_back(weight(newPartition.coordsForNode(*i)));
 *    }
 *
 *    double result = 0;
 *    unsigned step = 1;
 *
 *    double lastGain = expectedRunningTime(oldWeights, 0) 
 *        - expectedRunningTime(newWeights, 0);
 *    double residualGain = lastGain;
 *    double gain;
 *
 *    do  {
 *        gain = expectedRunningTime(oldWeights, step) 
 *            - expectedRunningTime(newWeights, step);
 *        residualGain = std::max(residualGain - std::abs(gain - lastGain), 0.0);
 *        lastGain = gain;
 *        step += 1;
 *        result += residualGain;
 *    } while ((residualGain > 0) && (step <= maxHorizon));
 *
 *    return result;
 *}
 */

/**
 * SIS
 */
/*
 *double LoadModel::expectedGainFromPartitioning(
 *    const Partition& oldPartition, 
 *    const Partition& newPartition,
 *    const unsigned& maxHorizon) const
 *{
 *    verifyMaster();
 *    if (!oldPartition.compatible(newPartition)) 
 *        throw std::invalid_argument("Partitions incompatible");
 *
 *    Nodes nodes = oldPartition.getNodes();
 *    DVec oldWeights;
 *    DVec newWeights;
 *    oldWeights.reserve(nodes.size());
 *    newWeights.reserve(nodes.size());
 *    for (Nodes::iterator i = nodes.begin(); i != nodes.end(); i++) {
 *        oldWeights.push_back(weight(oldPartition.coordsForNode(*i)));
 *        newWeights.push_back(weight(newPartition.coordsForNode(*i)));
 *    }
 *
 *    double lastGain = expectedRunningTime(oldWeights, 0) 
 *        - expectedRunningTime(newWeights, 0);
 *    return lastGain;
 *}
 */


/**
 * RSS (Remaining Steps Scaling, Buyyas algorithm)
 */
/*
 *double LoadModel::expectedGainFromPartitioning(
 *     const Partition& oldPartition, 
 *     const Partition& newPartition,
 *     const unsigned& maxHorizon) const
 *{
 * verifyMaster();
 * if (!oldPartition.compatible(newPartition)) 
 *     throw std::invalid_argument("Partitions incompatible");
 * 
 * Nodes nodes = oldPartition.getNodes();
 * DVec oldWeights;
 * DVec newWeights;
 * oldWeights.reserve(nodes.size());
 * newWeights.reserve(nodes.size());
 * for (Nodes::iterator i = nodes.begin(); i != nodes.end(); i++) {
 *     oldWeights.push_back(weight(oldPartition.coordsForNode(*i)));
 *     newWeights.push_back(weight(newPartition.coordsForNode(*i)));
 * }
 *
 * double lastGain = expectedRunningTime(oldWeights, 0) 
 *     - expectedRunningTime(newWeights, 0);
 * return lastGain * maxHorizon;
 *}
 */


/**
 * PLBb
 */
/*
 *double LoadModel::expectedGainFromPartitioning(
 *        const Partition& oldPartition, 
 *        const Partition& newPartition,
 *        const unsigned& maxHorizon) const
 *{
 *    verifyMaster();
 *    if (!oldPartition.compatible(newPartition)) 
 *        throw std::invalid_argument("Partitions incompatible");
 *    
 *    Nodes nodes = oldPartition.getNodes();
 *    DVec oldWeights;
 *    DVec newWeights;
 *    oldWeights.reserve(nodes.size());
 *    newWeights.reserve(nodes.size());
 *    for (Nodes::iterator i = nodes.begin(); i != nodes.end(); i++) {
 *        oldWeights.push_back(weight(oldPartition.coordsForNode(*i)));
 *        newWeights.push_back(weight(newPartition.coordsForNode(*i)));
 *    }
 *
 *    double result = 0;
 *    for (int step = 0; step < 4; step++)
 *        result += expectedRunningTime(oldWeights, step) 
 *            - expectedRunningTime(newWeights, step);
 *
 *    return result;
 *}
 */


/**
 * PLS
 */
double LoadModel::expectedGainFromPartitioning(
   const Partition& oldPartition, 
   const Partition& newPartition,
   const unsigned& maxHorizon) const
{
   verifyMaster();
   if (!oldPartition.compatible(newPartition)) 
       throw std::invalid_argument("Partitions incompatible");

   Nodes nodes = oldPartition.getNodes();
   DVec oldWeights;
   DVec newWeights;
   oldWeights.reserve(nodes.size());
   newWeights.reserve(nodes.size());
   for (Nodes::iterator i = nodes.begin(); i != nodes.end(); i++) {
       oldWeights.push_back(weight(oldPartition.coordsForNode(*i)));
       newWeights.push_back(weight(newPartition.coordsForNode(*i)));
   }

   double result = 0;
   unsigned step = 1;

   double lastGain = expectedRunningTime(oldWeights, 0) 
       - expectedRunningTime(newWeights, 0);
   double gain;
   double delta;
   DVec backGains;
   //backGains.push_back(lastGain);

   do  {
       gain = expectedRunningTime(oldWeights, step) 
           - expectedRunningTime(newWeights, step);
       delta = gain - lastGain;
       result += lastGain;
       lastGain = gain;
       //backGains.push_back(lastGain);
       step += 1;
   } while ((lastGain > 0) && (delta <= 0) && (step <= maxHorizon));

   //std::cout << "backGains: " << backGains << "\n";

   return result;
}


void LoadModel::verifyMaster() const
{
    unsigned rank = _mpilayer->rank();
    if (rank != _master) {
        throw std::logic_error(
                "Parent function may only be called by master (rank " 
                + StringConv::itoa(_master) + ") but was called by rank " + StringConv::itoa(rank));
    }
}


std::string LoadModel::timerSummary(
        const std::string& description, Chronometer& timer)
{
    long long cycleLength, workLength;
    timer.nextCycle(&cycleLength, &workLength);
    std::stringstream result;
    result
        << "LoadModel at rank " << _mpilayer->rank() << " spent " 
        << (double)workLength / (double)cycleLength
        << " of its time with " << description << ".\n";
    return result.str();
}


double LoadModel::predictRunningTime(
        const Partition& partition) const
{
    verifyMaster();
    Nodes nodes = partition.getNodes();
    if (nodes.size() != numNodes())
        throw std::invalid_argument("Partition size doesn't match.");

    DVec weights;
    weights.reserve(nodes.size());
    for (Nodes::iterator i = nodes.begin(); i != nodes.end(); i++) 
        weights.push_back(weight(partition.coordsForNode(*i)));

    return expectedRunningTime(weights, 0);
}


double LoadModel::expectedRunningTime(
    const DVec& nodeWeights, const unsigned& step) const
{
    double maxTime = 0;
    double time;
    for (size_t i = 0; i < nodeWeights.size(); i++) {
        time = nodeWeights.at(i) / powers(step)[i];
        maxTime = std::max(maxTime, time);
    }
    return maxTime;
}

};
#endif
