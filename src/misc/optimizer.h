#ifndef LIBGEODECOMP_MISC_OPTIMIZER_H
#define LIBGEODECOMP_MISC_OPTIMIZER_H

#include <libgeodecomp/misc/simulationparameters.h>
#include <limits>
namespace LibGeoDecomp {

class Optimizer
{
public:
    friend class OptimizerTest;

    class Evaluator
    {
    public:
        virtual ~Evaluator() {}
		Evaluator():calls(0){}
        virtual double operator()(SimulationParameters params) = 0;
		double getGlobalMax(){return maxima[0];}
		std::vector<double> getLocalMax(){return maxima;}
		int getCalls()const{return calls;}
		void resetCalls(){calls=0;}
	protected:
		int calls;
		std::vector<double> maxima; // first value is the global max
    };
	virtual ~Optimizer(){};
    explicit Optimizer(SimulationParameters params) :
        params(params),
        fitness(std::numeric_limits<double>::min())
    {}

    virtual SimulationParameters  operator()(int maxSteps, Evaluator& eval)
    {
        // fixme: this implementation is stupid!

        for (int i = 0; i < maxSteps; ++i) {
            SimulationParameters newParams = params;
            for (std::size_t i = 0; i < params.size(); ++i) {
                newParams[i] += ((rand() % 11) - 5) * newParams[i].getGranularity();
            }

            double newFitness = eval(newParams);

            if (newFitness > fitness) {
                params = newParams;
                fitness = newFitness;
            }
        }
		return params;
    }

protected:
    SimulationParameters params;
    double fitness;
};


}//namespace LibGeoDecomp



#endif
