#ifndef LIBGEODECOMP_MISC_OPTIMIZER_H
#define LIBGEODECOMP_MISC_OPTIMIZER_H

#include <libgeodecomp/misc/simulationparameters.h>

namespace LibGeoDecomp {

class Optimizer
{
public:
    friend class OptimizerTest;

    class Evaluator
    {
    public:
        virtual ~Evaluator()
        {}

        virtual double operator()(SimulationParameters params) = 0;
    };

    explicit Optimizer(SimulationParameters params) :
        params(params),
        fitness(std::numeric_limits<double>::min())
    {}

    void operator()(int maxSteps, Evaluator& eval)
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
    }

private:
    SimulationParameters params;
    double fitness;
};

}

#endif
