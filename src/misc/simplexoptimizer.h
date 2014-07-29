// vim: noai:ts=4:sw=4:expandtab
#ifndef LIBGEODECOMP_MISC_SIMPLEXOPTIMIZER_H
#define LIBGEODECOMP_MISC_SIMPLEXOPTIMIZER_H

// This is a Implementation of the siplex algorithm, dicribet in "Evolution and Optimum
// Seeking" written by Hans-Paul Schwefel.

#include <libgeodecomp/misc/optimizer.h>
#include <libgeodecomp/misc/simulationparameters.h>
#include <utility>

namespace LibGeoDecomp {

class SimplexOptimizer : public Optimizer
{
public:
    class SimplexVertex : public SimulationParameters
    {
    public:
        SimplexVertex(const SimulationParameters& point):
            SimulationParameters(point),
            fitness(-1)
        {}

        double getFitness() const
        {
            return fitness;
        }

        double evaluate(Evaluator& eval)
        {
            fitness = eval(*this);
            return fitness;
        }

        std::string toString() const;

        void setFitness(const double fitness)
        {
            this->fitness = fitness;
        }

    private:
        double fitness;
    }; //SimplexVertex

    SimplexOptimizer(const SimulationParameters& params);

    SimplexOptimizer(
        const SimulationParameters& params,
        const std::vector<double>& s,
        const double c,
        const double epsilon);

    virtual SimulationParameters operator()(int steps, Evaluator& eval);

private:
    void evalSimplex(Evaluator& eval);
    std::size_t minInSimplex();
    std::size_t maxInSimplex();
    void totalContraction();
    bool checkTermination();
    bool checkConvergence();
    std::pair<SimplexVertex, SimplexVertex> reflection();
    void initSimplex(SimulationParameters param);
    SimplexVertex expansion();
    SimplexVertex partialOutsideContraction();
    SimplexVertex partialInsideContraction();
    bool eq(vector<SimplexVertex> simplex1, vector<SimplexVertex> simplex2);
    std::vector<SimplexVertex> simplex;
    int comperator(double fitness);
    std::string simplexToString() const;
    std::vector<double> s;   // fixme: please rename this to "stepsizes"
    double c;   // fixme: documentation missing -- or better name should be found
    double epsilon;

    SimplexVertex merge(const SimplexVertex& a, const SimplexVertex& b) const;
};
// Caution: SimplexVertex have borders.
const SimplexOptimizer::SimplexVertex operator+(
        const SimplexOptimizer::SimplexVertex& a, const SimplexOptimizer::SimplexVertex& b);
const SimplexOptimizer::SimplexVertex operator+(
        const SimplexOptimizer::SimplexVertex& a, double b);
const SimplexOptimizer::SimplexVertex operator-(
        const SimplexOptimizer::SimplexVertex& a, const SimplexOptimizer::SimplexVertex& b);
const SimplexOptimizer::SimplexVertex operator*(
        const SimplexOptimizer::SimplexVertex& a, const SimplexOptimizer::SimplexVertex& b);
const SimplexOptimizer::SimplexVertex operator*(
        const SimplexOptimizer::SimplexVertex& a, const double& b);

} // namespace LibGeoDecomp

#endif //LIBGEODECOMP_MISC_SIMPLEXOPTIMIZER_H
