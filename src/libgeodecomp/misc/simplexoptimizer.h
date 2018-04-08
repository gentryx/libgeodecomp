#ifndef LIBGEODECOMP_MISC_SIMPLEXOPTIMIZER_H
#define LIBGEODECOMP_MISC_SIMPLEXOPTIMIZER_H

#include <libgeodecomp/misc/optimizer.h>
#include <libgeodecomp/misc/simulationparameters.h>

// Kill warning 4514 in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <utility>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4820 )
#endif

/**
 * This is a Implementation of the siplex algorithm, dicribet in
 * "Evolution and Optimum Seeking" written by Hans-Paul Schwefel.
 */
class SimplexOptimizer : public Optimizer
{
public:
    class SimplexVertex : public SimulationParameters
    {
    public:
        explicit
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

        void resetFitness()
        {
            this->fitness = -1;
        }

    private:
        double fitness;
    }; //SimplexVertex

    explicit
    SimplexOptimizer(
        const SimulationParameters& params,
        const double epsilon = -1.0,
        const double stepMultiplicator = 8.0,
        const std::vector<double>& stepsizes = std::vector<double>());

    virtual SimulationParameters operator()(unsigned steps, Evaluator& eval);

private:
    void evalSimplex(Evaluator& eval);
    std::size_t minInSimplex();
    std::size_t maxInSimplex();
    void totalContraction();
    bool checkConvergence();
    std::pair<SimplexVertex, SimplexVertex> reflection();
    void initSimplex(SimulationParameters newParams);
    SimplexVertex expansion();
    SimplexVertex partialOutsideContraction();
    SimplexVertex partialInsideContraction();
    bool eq(std::vector<SimplexVertex> simplex1, std::vector<SimplexVertex> simplex2);
    std::vector<SimplexVertex> simplex;
    int comperator(double fitness);
    std::string simplexToString() const;
    double epsilon;
    double stepMultiplicator;       // stepMultiplicator = c in Algo
    std::vector<double> stepsizes;  // stepsizes = s_i in Algo
    SimplexVertex merge(const SimplexVertex& a, const SimplexVertex& b) const;
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

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
        const SimplexOptimizer::SimplexVertex& a, double b);

} // namespace LibGeoDecomp

#endif //LIBGEODECOMP_MISC_SIMPLEXOPTIMIZER_H
