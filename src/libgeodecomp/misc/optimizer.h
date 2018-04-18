#ifndef LIBGEODECOMP_MISC_OPTIMIZER_H
#define LIBGEODECOMP_MISC_OPTIMIZER_H

#include <libgeodecomp/misc/limits.h>
#include <libgeodecomp/misc/simulationparameters.h>

namespace LibGeoDecomp {

// Hardwire this warning to off as MSVC would otherwise complain about
// inline functions not being included in object files -- and such.
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4710 4711 4820 )
#endif

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
        fitness(Limits<double>::getMin())
    {}

    virtual ~Optimizer()
    {}

    virtual SimulationParameters operator()(unsigned maxSteps, Evaluator& eval) = 0;

    double getFitness() const
    {
        return fitness;
    }

protected:
    SimulationParameters params;
    double fitness;
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif
