#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_partitioningsimulator_communicationmodel_h_
#define _libgeodecomp_parallelization_partitioningsimulator_communicationmodel_h_

#include <stdexcept>
#include <libgeodecomp/misc/commontypedefs.h>
#include <libgeodecomp/parallelization/partitioningsimulator/clustertable.h>
#include <libgeodecomp/parallelization/partitioningsimulator/nnlsfit.h>
#include <libgeodecomp/parallelization/partitioningsimulator/partition.h>

namespace LibGeoDecomp {

/**
 * CommunicationModel approximates the cost of changing the Partition.
 * Let p0 be the old Partition and p1 the new one.
 *
 * time(p0, p1) = a + b * maxChange(p0, p1) +  c * interClusterComm(p0, p1)
 *
 * maxChange(p0, p1) is the maximum of symmetric differences of regions assigned
 * to inidiviual nodes by p0 and p1.
 * This formula reflects the assumption that communication between the nodes
 * can be completely parallelized and the node with the highgest communication
 * requirements will dominated the total time
 *
 * interClusterComm(p0, p1) is the size of the regions that must be
 * reallocated across cluster borders.
 * This formula reflects the assumption that inter-cluster communication
 * is generally expensive.
 */
class CommunicationModel
{
    friend class CommunicationModelTest;

public:
    CommunicationModel(const ClusterTable& table);

    ~CommunicationModel();

    double predictRepartitionTime(
            const Partition& pOld, const Partition& pNew);

    void addObservation(
            const Partition& pOld, const Partition& pNew, const DVec& workLengths);

    /**
     * report on the current status
     */
    std::string report() const;

    /*
     * call this to get a final summary of CommunicationModel behavior
     */
    std::string summary();

private:
    NNLSFit* _fit;
    unsigned _numObservations;
    NNLSFit::DataPoints _observations;
    ClusterTable _table;

    static unsigned intersectedArea(
            const Partition& p0, const Partition& p1);

    static unsigned maxCommunication(
            const Partition& p0, const Partition& p1);

    static unsigned interClusterComm(
            const Partition& p0, const Partition& p1, const ClusterTable& table);

    DVec assembleParams(
        const Partition& p0, const Partition& p1) const;

    static void throwUnlessSizesMatch(const Partition& p0, const Partition& p1);
};

};

#endif
#endif
