#ifndef LEBGEODECOMP_MISC_PATTEROPTIMIZER_H
#define LEBGEODECOMP_MISC_PATTEROPTIMIZER_H

#include <libgeodecomp/misc/optimizer.h>
#include <libgeodecomp/misc/simulationparameters.h>
namespace LibGeoDecomp {


class PatternOptimizer: public Optimizer
{
public:
	explicit PatternOptimizer(SimulationParameters params);

	virtual void operator()(int steps, Evaluator& eval);
private:
	// TODO initiale stepwidth und min Stepwidth sollten automatisch aus der Dimensionsgröße generriert werden und optional von außen prametrisierbar sein.
	std::vector<double> stepwidth;
	std::vector<double> minStepwidth;
	bool reduceStepwidth();
	std::vector<SimulationParameters> genPattern(SimulationParameters middle);
	std::size_t getMaxPos(std::vector<SimulationParameters> pattern, Evaluator& eval);
	void printPattern(std::vector<SimulationParameters> pattern);
};




} // namespace LibGeoDecomp

#endif // LEBGEODECOMP_MISC_PATTEROPTIMIZER_H

