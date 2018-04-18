#ifndef LIBGEODECOMP_MISC_PATTERNOPTIMIZER_H
#define LIBGEODECOMP_MISC_PATTERNOPTIMIZER_H

#include <libgeodecomp/misc/optimizer.h>
#include <libgeodecomp/misc/simulationparameters.h>

namespace LibGeoDecomp {

#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4710 4711 )
#endif

/**
 * This class implements a direct search (pattern search) which
 * searches a given neighborhood to find better parameters.
 */
class PatternOptimizer : public Optimizer
{
public:
    explicit
    PatternOptimizer(
        SimulationParameters params,
        std::vector<double> stepwidth = std::vector<double>(),
        std::vector<double> minStepwidth = std::vector<double>());

    virtual SimulationParameters operator()(unsigned steps, Evaluator& eval);

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

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif


