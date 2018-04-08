#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/loadbalancer/oozebalancer.h>

// Kill warning 4514 in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <iostream>
#include <stdexcept>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

OozeBalancer::OozeBalancer(double newLoadWeight) :
    newLoadWeight(newLoadWeight)
{
    if (newLoadWeight < 0 || newLoadWeight > 1) {
        throw std::invalid_argument("bad loadWeight in OozeBalancer constructor");
    }
}


OozeBalancer::LoadVec OozeBalancer::expectedOptimalDistribution(
    const OozeBalancer::WeightVec& weights,
    const OozeBalancer::LoadVec& relativeLoads) const
{
    unsigned n = weights.size();
    // calculate approximate load share we want on each node
    double targetLoadPerNode = sum(relativeLoads) / n;

    if (targetLoadPerNode == 0) {
        return LoadVec(n, sum(weights)/ (double)n);
    }

    LoadVec ret(n, 0);
    LoadVec loadPerItem;
    for (unsigned i = 0; i < n; i++) {
        if (weights[i]) {
            LoadVec add(weights[i], relativeLoads[i] / weights[i]);
            append(loadPerItem, add);
        }
    }
    // stores the remaining fraction, which is still to be assigned,
    // of each item.
    LoadVec remFractPerItem(sum(weights), 1.0);

    // now fill up one node after another so that each gets his
    // targeted share...
    for (unsigned nodeC = 0; nodeC < n - 1; nodeC++) {
        double remLoad = targetLoadPerNode;

        for (unsigned itemC = 0; itemC < loadPerItem.size(); itemC++) {
            // skip already assigned items
            if (remFractPerItem[itemC] == 0) {
                continue;
            }

            // can we assign the whole remainder?
            double l = remFractPerItem[itemC] * loadPerItem[itemC];
            if (l <= remLoad) {
                ret[nodeC] += remFractPerItem[itemC];
                remFractPerItem[itemC] = 0;
                remLoad -= l;
            } else {
                double consumedFract = remLoad / loadPerItem[itemC];
                ret[nodeC] += consumedFract;
                remFractPerItem[itemC] -= consumedFract;
                remLoad = 0;
            }

            // break if node filled up
            if (remLoad == 0) {
                break;
            }
        }
    }

    // add remainder to last node
    ret.back() += sum(remFractPerItem);

    return ret;
}


OozeBalancer::WeightVec OozeBalancer::balance(
    const OozeBalancer::WeightVec& weights,
    const OozeBalancer::LoadVec& relativeLoads)
{
    LoadVec expectedOptimal = expectedOptimalDistribution(weights, relativeLoads);
    LoadVec newLoads = linearCombo(weights, expectedOptimal);

    WeightVec ret = equalize(newLoads);
    if (sum(weights) != sum(ret)) {
        LOG(FAULT, "OozeBalancer::balance() failed\n"
            << "  weights.sum() = " << sum(weights) << "\n"
            << "  ret.sum() = " << sum(ret) << "\n"
            << "  expectedOptimal.sum() = " << sum(expectedOptimal) << "\n"
            << "  weights = " << weights << "\n"
            << "  relativeLoads = " << relativeLoads << "\n"
            << "  expectedOptimal = " << expectedOptimal << "\n"
            << "  newLoads = " << newLoads << "\n"
            << "  ret = " << ret << "\n");
        throw std::logic_error("ret.sum does not match weights.sum");
    }
    return ret;
}


inline double frac(double d)
{
    double f = d - (long long)d;
    if (f < 0) {
        f = -f;
    }

    return f;
}


OozeBalancer::WeightVec OozeBalancer::equalize(const LoadVec& loads)
{
    OozeBalancer::WeightVec ret(loads.size());
    double balance = 0;
    for (unsigned i = 0; i < ret.size() - 1; i++) {
        double f = frac(loads[i]);
        double roundUpCost = 1 - f;
        ret[i] = static_cast<std::size_t>(loads[i]);

        if (roundUpCost < balance) {
            balance -= roundUpCost;
            ret[i]++;
        } else {
            balance += f;
        }
    }

    ret.back() = static_cast<std::size_t>(loads.back());

    if (frac(loads.back())  > (0.5 - balance)) {
        ret.back()++;
    }

    return ret;
}


OozeBalancer::LoadVec OozeBalancer::linearCombo(
    const OozeBalancer::WeightVec& oldLoads,
    const OozeBalancer::LoadVec& newLoads)
{
    OozeBalancer::LoadVec ret(newLoads.size());
    for (unsigned i = 0; i < ret.size(); i++) {
        ret[i] = newLoads[i] * newLoadWeight +
            oldLoads[i] * (1 - newLoadWeight);
    }

    return ret;
}

}

#ifdef _MSC_BUILD
#pragma warning( disable : 4710 )
#endif
