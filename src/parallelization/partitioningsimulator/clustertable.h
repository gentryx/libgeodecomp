#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_partitioningsimulator_clustertable_h_
#define _libgeodecomp_parallelization_partitioningsimulator_clustertable_h_

#include <libgeodecomp/misc/commontypedefs.h>
#include <libgeodecomp/mpilayer/mpilayer.h>
#include <libgeodecomp/parallelization/partitioningsimulator/nodes.h>

namespace LibGeoDecomp {

/**
 * manages cluster affiliation of nodes
 */
class ClusterTable : public SuperVector<Nodes>
{
    friend class ClusterTableTest;

public:
    /**
     * Single Cluster with nodes from 0 to size-1
     */
    ClusterTable(const unsigned& size);

    /**
     * @a node2cluster : node2cluster[node] = cluster.  if cluster < 0, the node
     * will not be added to the table
     */
    ClusterTable(const IVec& node2cluster);

    /**
     * constructs ClusterTable from a cluster file by calling hostname on every
     * node and syncing the information
     */
    ClusterTable(MPILayer& mpilayer, const std::string& clusterConfig);

    bool sameCluster(const Nodes& nodes) const;

    /**
     * reorders clusters in descending size
     */
    void sort();

    /**
     * returns a ClusterTable that conforms to the original but only contains @a
     * nodes
     */
    ClusterTable subSet(const Nodes& nodes) const;

    bool operator==(const ClusterTable& other) const;
    bool operator!=(const ClusterTable& other) const;

    std::string toString() const;

    unsigned numNodes() const;

    UVec sizes() const;

private:
    ClusterTable() {};

    void init(const IVec& node2cluster);

    int readClusterFromFile(
        const std::string& filename, const std::string& hostname);
};

};

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const LibGeoDecomp::ClusterTable& table)
{
    __os << table.toString();
    return __os;
}

#endif
#endif
