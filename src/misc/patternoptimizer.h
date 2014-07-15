// vim: noai:ts=4:sw=4:expandtab
#ifndef LEBGEODECOMP_MISC_PATTEROPTIMIZER_H
#define LEBGEODECOMP_MISC_PATTEROPTIMIZER_H

#include <libgeodecomp/misc/optimizer.h>
#include <libgeodecomp/misc/simulationparameters.h>
#include <string>
namespace LibGeoDecomp {

class PatternOptimizer: public Optimizer
{
public:
    friend class PatternOptimizerTest;
    explicit PatternOptimizer(SimulationParameters params);
    virtual SimulationParameters operator()(int steps, Evaluator& eval);
private:
    // TODO initiale stepwidth und min Stepwidth sollten automatisch aus der Dimensionsgröße generriert werden und optional von außen prametrisierbar sein.
    int stepFaktor;
    int maxSteps;
    std::vector<double> stepwidth;
    std::vector<double> minStepwidth;
    bool reduceStepwidth();
    std::vector<SimulationParameters> genPattern(SimulationParameters middle);
    std::size_t getMaxPos(std::vector<SimulationParameters> pattern, Evaluator& eval, std::size_t oldMiddle);
    std::string printPattern(std::vector<SimulationParameters> pattern);
};
} // namespace LibGeoDecomp

#endif // LEBGEODECOMP_MISC_PATTEROPTIMIZER_H

