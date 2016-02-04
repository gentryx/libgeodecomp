#ifndef LIBGEODECOMP_GEOMETRY_PARTITIONS_CHECKERBOARDINGPARTITION_H
#define LIBGEODECOMP_GEOMETRY_PARTITIONS_CHECKERBOARDINGPARTITION_H

#include <libgeodecomp/geometry/partitions/partition.h>

namespace LibGeoDecomp {

/**
 * One of the most used decompostition techniques in computer
 * simulations. It yields cuboid subdomains, but can't handle load
 * balancing (neither static nor dynamic). General advice is to use
 * the RecursiveBisectionPartition or the ZCurvePartition instead.
 */
template<int DIM>
class CheckerboardingPartition : public Partition<DIM>
{
public:
    using Partition<DIM>::startOffsets;
    using Partition<DIM>::weights;

    inline explicit CheckerboardingPartition(
        const Coord<DIM>& origin = Coord<DIM>(),
        const Coord<DIM>& dimensions = Coord<DIM>(),
        const long& offset = 0,
        const std::vector<std::size_t>& weights = std::vector<std::size_t>(2)) :
        Partition<DIM>(offset, weights),
        origin(origin),
        dimensions(dimensions)
    {
        nodeGridDim = getNodeGridDim(weights.size());
    }

    Region<DIM> getRegion(const std::size_t node) const
    {
        Coord<DIM> logicalCoord(node % nodeGridDim.x(),
                               (node % (nodeGridDim.x() * nodeGridDim.y()))/ nodeGridDim.x());
        if (DIM > 2){
            logicalCoord[2] = node / (nodeGridDim.x() * nodeGridDim.y());
        }

        Coord<DIM> realStart;
        Coord<DIM> realEnd;
        for(int i = 0; i < DIM; ++i){
            realStart[i] = logicalCoord[i] * dimensions[i] / nodeGridDim[i];
            realEnd[i] = (logicalCoord[i]+1) * dimensions[i] / nodeGridDim[i];
        }
        Region<DIM> r;
        r << CoordBox<DIM>(origin + realStart, realEnd - realStart);
        return r;
    }

private:
    Coord<DIM> origin;
    Coord<DIM> dimensions;
    Coord<DIM> nodeGridDim;

    Coord<DIM> getNodeGridDim(const std::size_t totalNodes) const
    {
        std::size_t remainingNodes = totalNodes;
        std::size_t limit = sqrt(remainingNodes);
        std::vector<std::size_t> primes = primeFactors(limit);
        std::vector<std::pair<std::size_t, int> > factors;

        for (std::vector<std::size_t>::reverse_iterator i = primes.rbegin();
             i != primes.rend();
             ++i) {
            int occurences = 0;
            while ((remainingNodes % *i) == 0) {
                ++occurences;
                remainingNodes /= *i;
            }

            if (occurences > 0) {
                factors.push_back(std::make_pair(*i, occurences));
            }
        }

        if (remainingNodes != 1) {
            push_front(factors, std::make_pair(remainingNodes, 1));
        }

        Coord<DIM> ret = Coord<DIM>::diagonal(1);

        for (std::vector<std::pair<std::size_t, int> >::iterator i = factors.begin();
             i != factors.end();
             ++i) {
            std::size_t prime = i->first;
            int occurences = i->second;

            for (int i = 0; i < occurences; ++i) {
                *minElement(ret) *= prime;
            }
        }

        std::sort(&ret[0], &ret[0] + DIM);
        return ret;
    }

    inline int *minElement(Coord<DIM>& coord) const
    {
        int *ret = &coord[0];

        for (int i = 1; i < DIM; ++i) {
            if (coord[i] < *ret) {
                ret = &coord[0] + i;
            }
        }

        return ret;
    }

    inline std::vector<std::size_t> primeFactors(std::size_t limit) const
    {
        std::vector<std::size_t> primes;

        primes.push_back(2);

        for (std::size_t i = 3; i <= limit; i += 2) {
            std::vector<std::size_t>::iterator iter = primes.begin();
            for (; iter != primes.end(); ++iter) {
                if ((i % *iter) == 0) {
                    break;
                }
            }

            if (iter == primes.end()) {
                primes.push_back(i);
            }
        }

        return primes;
    }
};

}

#endif
