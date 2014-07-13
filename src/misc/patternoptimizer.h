#ifndef LEBGEODECOMP_MISC_PATTEROPTIMIZER_H
#define LEBGEODECOMP_MISC_PATTEROPTIMIZER_H

#include <libgeodecomp/misc/optimizer.h>
//#include <libgeodecomp/misc/simulationparameters.h>
#include <cmath>

//#define MATHIAS_DBG

#define STEP_FACTOR 6
#define MAX_STEPS 8
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

PatternOptimizer::PatternOptimizer(SimulationParameters params):
	Optimizer(params)//,
{
	for(std::size_t i = 0; i < params.size();++i) {
		double dimsize = Optimizer::params[i].getMax()
			- Optimizer::params[i].getMin();
		stepwidth.push_back(dimsize/STEP_FACTOR);
		double tmp = stepwidth[i]/std::pow(2,MAX_STEPS);
		// wenn die minStepwidht kleiner ist als die Granularity 
		// kann diese nicht mehr als Abbruchkriterium dienen, deshalb 
		// minStepwidth >= Granularity
		if(tmp < params[i].getGranularity())
			tmp=params[i].getGranularity();
		minStepwidth.push_back(tmp);
	}
}

// wenn alle parameter am minimum sind wird false zurueck gegeben
bool PatternOptimizer::reduceStepwidth() 			{
#ifdef MATHIAS_DBG
	std::cout<< "reduceStepwidth:" << std::endl;
#endif // MATHIAS_DBG	

	bool allWasMin =true;  
	for(size_t i = 0;i < stepwidth.size(); ++i)
	{
#ifdef MATHIAS_DBG
		std::cout<< "Dimension "<< i << ": " << stepwidth[i];
#endif // MATHIAS_DBG
		if(stepwidth[i] <= minStepwidth[i])
		{
			stepwidth[i] = minStepwidth[i];
			continue;
		}
		stepwidth[i]=stepwidth[i] / 2;
		if(stepwidth[i]<= minStepwidth[i])
			stepwidth[i] = minStepwidth[i];
		allWasMin = false;
#ifdef MATHIAS_DBG
		std::cout<< " --> " << stepwidth[i]<< std::endl;
#endif // MATHIAS_DBG
	}
	return !allWasMin;
}


std::vector<SimulationParameters> PatternOptimizer::genPattern(SimulationParameters middle)
	{
		std::vector<SimulationParameters> result(middle.size()*2+1);
		result[0]=middle;
		for(std::size_t i = 0; i < middle.size(); ++i)
		{
			SimulationParameters tmp1(middle),tmp2(middle);
			tmp1[i]+= stepwidth[i];
			tmp2[i]+= (stepwidth[i] * -1.0);
			result[1+i*2] = tmp1;
			result[2+i*2] = tmp2;
		}	
		return result;
	}

std::size_t PatternOptimizer::getMaxPos(std::vector<SimulationParameters> pattern, Evaluator& eval)
{	
	std::size_t retval=-1;
	double newFitness;
	for(std::size_t i = 0; i< pattern.size(); ++i)
	{
		newFitness = eval(pattern[i]);
		if(newFitness>=Optimizer::fitness)
		{
			retval = i;
			Optimizer::fitness = newFitness;
		}
	}
	return retval;
}

void PatternOptimizer::operator()(int steps,Evaluator & eval)
{
	
	SimulationParameters middle =Optimizer::params;
	
	std::vector<SimulationParameters> pattern = genPattern(middle);
	for (int k = 0; k< 100; ++k)
	{
		pattern = genPattern(middle);
#ifdef MATHIAS_DBG
		printPattern(pattern);
#endif // MATHIAS_DBG
		std::size_t maxPos = getMaxPos(pattern, eval);
		if(maxPos == 0)			// center was the Maximum
		{
			if(!reduceStepwidth())	// abort test
				break;
		}else{					// do next step
			middle = pattern[maxPos];
		}
	}
}
void PatternOptimizer::printPattern(std::vector<SimulationParameters> pattern){
	std::cout<< std::endl;
	for(std::size_t i = 0; i<pattern.size(); ++i){
		if(i==0)
			std::cout<< "Middle:		";
		else
			std::cout<< "Direction: " << i<< " :	";
		for(std::size_t j = 0; j < pattern[i].size(); ++j)
			std::cout <<  pattern[i][j].getValue()<< ", " <<pattern[i][j].getValue() << " - ";
		std::cout<< std::endl;
	}
	std::cout<< std::endl;
}



} // namespace LibGeoDecomp

#endif // LEBGEODECOMP_MISC_PATTEROPTIMIZER_H

