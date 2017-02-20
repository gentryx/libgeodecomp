// vim: noai:ts=4:sw=4:expandtab
#ifndef LIBGEODECOMP_MISC_OPTIMIZER_H
#define LIBGEODECOMP_MISC_OPTIMIZER_H

#include <libgeodecomp/misc/simulationparameters.h>
#include <limits>

namespace LibGeoDecomp {

/**
 * An Optimizer's purpose is to drive generic parameters so that the
 * fitness function will be maximized.
 */
class Optimizer
{
public:
    friend class OptimizerTest;

    class Evaluator
    {
    public:
        virtual ~Evaluator()
        {}

        virtual double operator()(const SimulationParameters& params) = 0;
    };

    explicit Optimizer(SimulationParameters params) :
        params(params),
        fitness(std::numeric_limits<double>::min())
    {}

    virtual ~Optimizer()
    {}

    virtual SimulationParameters operator()(int maxSteps, Evaluator& eval) = 0;

    double getFitness() const
    {
        return fitness;
    }

protected:
    SimulationParameters params;
    double fitness;
};

}



#endif
