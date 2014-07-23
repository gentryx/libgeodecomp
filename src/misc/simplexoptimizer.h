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
    friend class SimplexOptimizerTest;
    
    class SimplexVertex : public SimulationParameters
    {
    public:
        SimplexVertex(SimulationParameters point):
            SimulationParameters(point),
            fitness(-1)
        {}
        double getFitness()
        {
            return fitness;
        }
        double evaluate(Evaluator& eval)
        {   
            fitness = eval(*this);
            return fitness;
        }
        std::string toString();
        void setFitness(double fitness)
        {
            this->fitness = fitness;
        }
    private:
        double fitness;
    }; //SimplexVertex
    SimplexOptimizer(SimulationParameters params);
    SimplexOptimizer(SimulationParameters params, std::vector<double> s, double c, double epsilon);
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
    std::string simplexToString();
    std::vector<double> s;   // Stepsize
    double c;   // 
    double epsilon;
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
