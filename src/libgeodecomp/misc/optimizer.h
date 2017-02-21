/**
 * Copyright 2014-2017 Andreas Schäfer
 * Copyright 2014 Mathias Schöll
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef LIBGEODECOMP_MISC_OPTIMIZER_H
#define LIBGEODECOMP_MISC_OPTIMIZER_H

#include <libgeodecomp/misc/limits.h>
#include <libgeodecomp/misc/simulationparameters.h>

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
        fitness(Limits<double>::getMin())
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
