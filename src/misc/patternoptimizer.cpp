#include <libgeodecomp/misc/patternoptimizer.h>
#include <cmath>


//#define MATHIAS_DBG

#define STEP_FACTOR 2 
#define MAX_STEPS 8
namespace LibGeoDecomp{

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
		std::cout<< " --> " << stepwidth[i]<< "; "<< std::endl;
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
	std::size_t retval=0;
	double newFitness;
	for(std::size_t i = 1; i< pattern.size(); ++i) //i=1 weil die mitte im schritt zuvor schon berechnet wurde und fitness bekannt ist
	{
		newFitness = eval(pattern[i]);
		if(newFitness>Optimizer::fitness)
		{
			retval = i;
			Optimizer::fitness = newFitness;
		}
	}
	return retval;
}

SimulationParameters PatternOptimizer::operator()(int steps,Evaluator & eval)
{
	
	SimulationParameters middle =Optimizer::params;
	
	std::vector<SimulationParameters> pattern = genPattern(middle);
	for (int k = 0; k< steps; ++k)
	{
		pattern = genPattern(middle);
#ifdef MATHIAS_DBG
		printPattern(pattern);
#endif // MATHIAS_DBG
		std::size_t maxPos = getMaxPos(pattern, eval);
		if(maxPos == 0)			// center was the Maximum
		{
			if(!reduceStepwidth())	// abort test
				return middle;
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
			std::cout <<  pattern[i][j].getValue()<< " - ";
		std::cout<< std::endl;
	}
	std::cout<< std::endl;
}
} //namespace LibGeoDecomp
