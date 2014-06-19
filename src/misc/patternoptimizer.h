#ifndef LEBGEODECOMP_MISC_PATTEROPTIMIZER_H
#define LEBGEODECOMP_MISC_PATTEROPTIMIZER_H

#include <libgeodecomp/misc/optimizer.h>
#include <libgeodecomp/misc/simulationparameters.h>
#include <cmath>


#define STEP_FACTOR 6
#define MAX_STEPS 8
namespace LibGeoDecomp {
class PatternOptimizer: public Optimizer
{
public:
	// TODO muss die nested Class Evaluator neu angelegt werden oder erbe ich diese mit???

	// expicit von Optimizer übernommen warum brauchen wir das hier?
	explicit PatternOptimizer(SimulationParameters params);

	virtual void operator()(Evaluator& eval);
private:
	// TODO initiale stepwidth und min Stepwidth sollten automatisch aus der Dimensionsgröße generriert werden und optional von außen prametrisierbar sein.
	std::vector<double> stepwidth;
	std::vector<double> minStepwidth
	bool reduceStepwidth();
	std::vector<SimulationParameters> genPattern(SimulationParameters middle);
	std::size_t getMaxPos(std::vector<SimulationParameters> pattern, Evaluator& eval)
};

explicit PatternOptimizer::PatternOptimizer(SimulationParameters params):
	SimulationParameters(parameters),
	stepwidth(std::vector<double>stepwidth(params.size(),0.0)),
	minStepwidth(std::vector<double>minStepwidth(params.size(),0.0))
{
	for(std::size_t i = 0; i < stepwidth.size();++i) {
		// stepwith.size() == minStepwidth.size() == params.size() 
		double dimsize = SimulationParameters::params[i].getMin()
			- SimulationParameters::params[i]getMax();
		stepwidth[i]= dimsize / STEP_FACTOR;// to rounding is parameters job!!!
		minStepwidth[i]=stepwidth[i] / std::pow(2, MAX_STEPS);

	}
}

// wenn alle parameter am minimum sind wird false zurueck gegeben
bool PatternOptimicer::reduceStepwidth() 			{
	bool allWasMin =true;  
	for(size_t i = 0;i < stepwidht.size(); ++i)
	{
		if(stepwidth[i] <= minStepwidth[i])
		{
			stepwidht[i] = minStepwidth[i];
			continue;
		}
		stepwidth[i]=stepwidth[i] / 2;
		if(stepwidth[i]<= min Stepwidth[i])
			stepwidth[i] = minStepwidth[i];
		allWasMin = false;
	}
	return !allWasMin;
}


std::vector<SimulationParameters> PatternOptimizer::genPattern(SimulationParameters middle)
	{
		std::vector<SimulationParameters> result(middle.size()*2+1);
		result[0]=middle;
		for(std::size_t i = 0; i < middle.size; ++i)
		{
			SimulationParameters tmp1(middle),tmp2(middle);
			tmp1[i]+= stepwidth[i];
			tmp2[i]+= stepwidth[i] * -1;
			result[1+i*2] = tmp1;
			tesult[2+i*2] = tmp2;
		}	

	}

std::size_t PatternOptimizer::getMaxPos(std::vector<SimulationParameters> pattern, Evaluator& eval)
{
	std::size_t retval=-1;
	double newFitness;
	for(std::size_t i = 0; i< pattern.size(); ++i)
	{
		newFitness = eval(pattern[i]);
		if(newFitness>Optimizer::fitness)
		{
			retval = i;
			fitness = newFitness;
		}
	}
	return reval;
}

void PatternOptimizer::operator()(Evaluator & eval)
{
	SimulationParameters middle = Optimizer::params;
	for (;;)
	{
		std::vector<SimulationParameters> pattern = genPattern(middle);
		std::size_t maxPos = getMaxPos(pattern, eval);
		if(maxPos == 0)			//center was the Maximum
		{
			if(!reduzeStepwidth(middle))	// abort criterion
				continue;
		}else{
			middle = pattern[maxPos];
		}
	}
}

} // namespace LibGeoDecomp

#endif // LEBGEODECOMP_MISC_PATTEROPTIMIZER_H

