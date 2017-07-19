#include <libgeodecomp/loadbalancer/loadbalancer.h>

namespace LibGeoDecomp {

std::vector<std::size_t> LoadBalancer::initialWeights(std::size_t items, const std::vector<double>& rankSpeeds)
{
    std::size_t size = rankSpeeds.size();
    double totalSum = sum(rankSpeeds);
    std::vector<std::size_t> ret(size);

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

