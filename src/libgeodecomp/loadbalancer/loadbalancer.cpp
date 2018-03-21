#include <libgeodecomp/loadbalancer/loadbalancer.h>
#include <stdexcept>

namespace LibGeoDecomp {

LoadBalancer::WeightVec LoadBalancer::initialWeights(std::size_t items, const LoadBalancer::LoadVec& rankSpeeds)
{
    std::size_t size = rankSpeeds.size();
    if (size == 0) {
        throw std::invalid_argument("Can't gather weights for 0 nodes.");
    }

    double totalSum = sum(rankSpeeds);
    LoadBalancer::WeightVec ret(size);

    std::size_t lastPos = 0;
    double partialSum = 0.0;
    for (std::size_t i = 0; i < size - 1; ++i) {
        partialSum += rankSpeeds[i];
        std::size_t nextPos = items * partialSum / totalSum;
        ret[i] = nextPos - lastPos;
        lastPos = nextPos;
    }
    ret[size - 1] = items - lastPos;

    return ret;
}

}

