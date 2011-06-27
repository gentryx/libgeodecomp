#include <iostream>
#include <stdexcept>
#include <libgeodecomp/loadbalancer/oozebalancer.h>

namespace LibGeoDecomp {

OozeBalancer::OozeBalancer(double newLoadWeight) : _newLoadWeight(newLoadWeight) 
{
    if (newLoadWeight < 0 || newLoadWeight > 1) {
        throw std::invalid_argument(
                "bad loadWeight in OozeBalancer constructor");
    }
}


DVec OozeBalancer::expectedOptimalDistribution(const UVec& currentLoads, const DVec& relativeLoads) const
{
    unsigned n = currentLoads.size();
    // calculate approximate load share we want on each node
    double targetLoadPerNode = relativeLoads.sum() / n;

    if (targetLoadPerNode == 0) 
        return DVec(n, currentLoads.sum()/ (double)n);    

    DVec ret(n, 0);
    DVec loadPerItem;   
    for (unsigned i = 0; i < n; i++) {
        if (currentLoads[i]) {
            DVec add(currentLoads[i], relativeLoads[i] / currentLoads[i]);
            loadPerItem.append(add);
        }
    }
    // stores the remaining fraction, which is still to be assigned,
    // of each item.
    DVec remFractPerItem(currentLoads.sum(), 1.0);

    // now fill up one node after another so that each gets his
    // targeted share...
    for (unsigned nodeC = 0; nodeC < n - 1; nodeC++) {
        double remLoad = targetLoadPerNode;
        for (unsigned itemC = 0; itemC < loadPerItem.size(); itemC++) {
            // skip already assigned items
            if (remFractPerItem[itemC] == 0) continue;
            
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
            if (remLoad == 0) break;
        }
    }

    // add remainder to last node
    ret.back() += remFractPerItem.sum();

    return ret;
}


UVec OozeBalancer::balance(const UVec& currentLoads, const DVec& relativeLoads)
{    
    DVec expectedOptimal = expectedOptimalDistribution(currentLoads, relativeLoads);
    DVec newLoads = linearCombo(currentLoads, expectedOptimal);

    // I don't trust this code so much, yet.
    UVec ret = equalize(newLoads);
    if (currentLoads.sum() != ret.sum()) {
        std::cerr << "OozeBalancer::balance() failed\n"            
                  << "  currentLoads.sum() = " << currentLoads.sum() << "\n"
                  << "  ret.sum() = " << ret.sum() << "\n"
                  << "  expectedOptimal.sum() = " << expectedOptimal.sum() << "\n"
                  << "  currentLoads = " << currentLoads << "\n"
                  << "  relativeLoads = " << relativeLoads << "\n"
                  << "  expectedOptimal = " << expectedOptimal << "\n"
                  << "  newLoads = " << newLoads << "\n"
                  << "  ret = " << ret << "\n"
                  << "IMPORTANT: please send a mail with this log attached to Andreas Schäfer, gentryx@gmx.de\n";
        throw std::logic_error("ret.sum does not match currentLoads.sum");
    }
    return ret;
}


inline double frac(const double& d)
{
    double f = d - (long long)d;
    if (f < 0) f = -f;
    return f;
}


UVec OozeBalancer::equalize(const DVec& loads)
{
    UVec ret(loads.size());
    double balance = 0;
    for (unsigned i = 0; i < ret.size() - 1; i++) {
        double f = frac(loads[i]);
        double roundUpCost = 1 - f;
        ret[i] = (int)loads[i];

        if (roundUpCost < balance) {
            balance -= roundUpCost;
            ret[i]++;
        } else {
            balance += f;
        }            
    }
    
    ret.back() = (int)loads.back();
    if (frac(loads.back())  > (0.5 - balance))
        ret.back()++;

    return ret;
}


DVec OozeBalancer::linearCombo(const UVec& oldLoads, const DVec& newLoads)
{
    DVec ret(newLoads.size());
    for (unsigned i = 0; i < ret.size(); i++)
        ret[i] = newLoads[i] * _newLoadWeight + 
            oldLoads[i] * (1 - _newLoadWeight);
    return ret;
}

};
