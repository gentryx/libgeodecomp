// vim: noai:ts=4:sw=4:expandtab
#ifndef LIBGEODECOMP_MISC_PATTERNOPTIMIZER_H
#define LIBGEODECOMP_MISC_PATTERNOPTIMIZER_H

#include <libgeodecomp/misc/optimizer.h>
#include <libgeodecomp/misc/simulationparameters.h>

namespace LibGeoDecomp {

class PatternOptimizer : public Optimizer
{
public:
    explicit
    PatternOptimizer(
        SimulationParameters params,
        std::vector<double> stepwidth = std::vector<double>(),
        std::vector<double> minStepwidth = std::vector<double>());

    virtual SimulationParameters operator()(int steps, Evaluator& eval);

private:
    std::vector<double> stepwidth;
    std::vector<double> minStepwidth;

    bool reduceStepwidth();

    std::vector<SimulationParameters> genPattern(SimulationParameters middle);

    std::size_t getMaxPos(
        const std::vector<SimulationParameters>& pattern,
        Evaluator& eval,
        std::size_t oldMiddle);

    std::string patternToString(std::vector<SimulationParameters> pattern);
};
}

#endif


