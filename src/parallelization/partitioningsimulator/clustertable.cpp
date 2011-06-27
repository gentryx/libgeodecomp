#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#include <stdexcept>
#include <algorithm>
#include <cerrno>
#include <unistd.h>
#include <libgeodecomp/misc/exceptions.h>
#include <libgeodecomp/misc/superset.h>
#include <libgeodecomp/misc/supermap.h>
#include <libgeodecomp/parallelization/partitioningsimulator/clustertable.h>

namespace LibGeoDecomp {

ClusterTable::ClusterTable(const unsigned& size)
{
    if (size > 0) push_back(Nodes::firstNum(size));
}


ClusterTable::ClusterTable(const IVec& node2cluster)
{
    init(node2cluster);
}


ClusterTable::ClusterTable(MPILayer& mpilayer, const std::string& clusterConfig)
{
    const int HOST_NAME_MAX = 1024;
    char hostname[HOST_NAME_MAX];
    int err = gethostname(hostname, HOST_NAME_MAX);
    
    if (err != 0) {
        throw std::invalid_argument("ClusterTable: \
            host does not provide HOSTNAME");
    }
    int localCluster = readClusterFromFile(
        clusterConfig, std::string(hostname));
    init(mpilayer.allGather(localCluster));
}


void ClusterTable::init(const IVec& node2cluster)
{
    // we want clusters with consecutive numbers starting at 0. so we may need
    // to re-index
    SuperSet<int> clusters;
    for (IVec::const_iterator i = node2cluster.begin(); 
        i != node2cluster.end(); i++) 
        clusters.insert(*i);

    resize(clusters.size(), Nodes());
    SuperMap<int, unsigned> reIndexed;

    unsigned index = 0;
    for (SuperSet<int>::const_iterator i = clusters.begin();
        i != clusters.end(); i++) {
        reIndexed[*i] = index;
        index++;
    }

    for (unsigned i = 0; i < node2cluster.size(); i++) 
        (*this)[reIndexed[node2cluster[i]]].insert(i);
}


bool ClusterTable::sameCluster(const Nodes& nodes) const
{
    return subSet(nodes).size() <= 1;
}


bool ClusterTable::operator==(const ClusterTable& other) const
{
    return SuperVector<Nodes>::operator==(other);
}


bool ClusterTable::operator!=(const ClusterTable& other) const
{
    return !(*this == other);
}


bool clusterCompareBySize(const Nodes& n0, const Nodes& n1)
{
    return n0.size() > n1.size();
}


void ClusterTable::sort()
{
    std::sort(this->begin(), this->end(), clusterCompareBySize);
}


ClusterTable ClusterTable::subSet(const Nodes& nodes) const
{
    ClusterTable result;
    for (unsigned i = 0; i < size(); i++) {
        Nodes intersection = at(i) && nodes;
        if (intersection.size() > 0) result.push_back(intersection);
    }
    return result;
}


std::string ClusterTable::toString() const
{
    std::stringstream s;
    s << SuperVector<Nodes>::toString();
    return s.str();
}    


unsigned ClusterTable::numNodes() const
{
    return sizes().sum();
}


UVec ClusterTable::sizes() const
{
    UVec result(size());
    for (unsigned i = 0; i < size(); i ++) result[i] = at(i).size();
    return result;
}


int ClusterTable::readClusterFromFile(
    const std::string&, const std::string&)
{
    // fixme
    return 0;
}

// int ClusterTable::readClusterFromFile(
//     const std::string& filename, const std::string& hostname)
// {
//     libconfig::Config config;
//     try {
//         config.loadFile(filename.c_str());
//     } catch (libconfig::ParseException& e) {
//         std::ostringstream errmsg;
//         errmsg << "Error while parsing ";
//         errmsg << filename << ":" << e.getLine() << " " << e.getError();
//         throw std::runtime_error(errmsg.str());
//     } catch (libconfig::FileIOException& e) {
//         std::ostringstream errmsg;
//         errmsg << "cannot open config file `";
//         errmsg << filename << "': " << strerror(errno);
//         throw std::runtime_error(errmsg.str());
//     }

//     try {
//         return config.lookup(hostname);
//     } catch (libconfig::SettingNotFoundException) {
//         throw UsageException(
//             "Error while parsing config file: missing hostname `" 
//             + hostname + "'");
//     }
// }

};

#endif
