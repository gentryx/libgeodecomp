// Kill warning 4514 in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/misc/simplexoptimizer.h>
#include <libgeodecomp/misc/limits.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp{


//------------------------ simplexVertex -------------------------

std::string SimplexOptimizer::SimplexVertex::toString() const
{
    std::stringstream result;
    result << std::endl;
    for (std::size_t j = 0;j < this->size(); ++j) {
        result << this->operator[](j).getValue() << "; ";
    }
    result << "fitness: " << getFitness();
    result << std::endl;
    return result.str();
}

const SimplexOptimizer::SimplexVertex operator+(
        const SimplexOptimizer::SimplexVertex& a,
        const SimplexOptimizer::SimplexVertex& b)
{
    SimplexOptimizer::SimplexVertex result(a);
    if (a.size() == b.size()){
        for (std::size_t i = 0; i < b.size(); ++i) {
            result[i].setValue(a.operator[](i).getValue() + b[i].getValue());
        }
        result.resetFitness();
        return result;
    }

    throw std::invalid_argument("different size of SimplexVertex in operator+ call");
}

const SimplexOptimizer::SimplexVertex operator+(
         const SimplexOptimizer::SimplexVertex& a,
         double b)
{
    SimplexOptimizer::SimplexVertex result(a);
    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i].setValue(a.operator[](i).getValue() + b);
    }
    result.resetFitness();
    return result;
}

const SimplexOptimizer::SimplexVertex operator-(
        const SimplexOptimizer::SimplexVertex& a,
        const SimplexOptimizer::SimplexVertex& b)
{
    SimplexOptimizer::SimplexVertex result(a);
    if(a.size() == b.size()) {
        for (std::size_t i = 0; i < b.size(); ++i) {
            result[i].setValue(a.operator[](i).getValue() - b[i].getValue());
        }
        result.resetFitness();
        return result;
    }

    throw std::invalid_argument("different size of SimplexVertex in operator- call");
}

const SimplexOptimizer::SimplexVertex operator*(
        const SimplexOptimizer::SimplexVertex& a,
        const SimplexOptimizer::SimplexVertex& b)
{
    SimplexOptimizer::SimplexVertex result(a);
    if (a.size()==b.size()){
        for (std::size_t i = 0; i < b.size(); ++i) {
            result[i].setValue(a.operator[](i).getValue() * b[i].getValue());
        }
        result.resetFitness();
        return result;
    }

    throw std::invalid_argument("different size of SimplexVertex in operator* call");
}

const SimplexOptimizer::SimplexVertex operator*(
        const SimplexOptimizer::SimplexVertex& a,
        double b)
{
    SimplexOptimizer::SimplexVertex result(a);
    for (std::size_t i = 0; i < a.size(); ++i) {
        result[i].setValue(a.operator[](i).getValue() * b);
    }
    result.resetFitness();
    return result;
}

//------------------------simplexOptimizer-------------------------

SimplexOptimizer::SimplexOptimizer(
        const SimulationParameters& params,
        const double epsilon,
        const double stepMultiplicator,
        const std::vector<double>& stepsizes) :
    Optimizer(params),
    epsilon(epsilon),
    stepMultiplicator(stepMultiplicator),
    stepsizes(stepsizes)
{
    LOG(Logger::INFO, "this->epsilon: " << this->epsilon << " this->stepMultiplicator: " << this->stepMultiplicator)
    if (stepsizes.size() == 0) {
        for (std::size_t i = 0; i < params.size(); ++i) {
           if (params[i].getGranularity() > 1) {
               this->stepsizes.push_back(params[i].getGranularity());
           } else {
               this->stepsizes.push_back(1);
           }
        }
    }
    if (this->stepsizes.size() != params.size()) {
        throw std::invalid_argument("stepsize.size() =! params.size()");
    }
    initSimplex(params);
}

SimulationParameters SimplexOptimizer::operator()(unsigned steps, Evaluator& eval)
{
    evalSimplex(eval);
    unsigned i;
    for (i = 0; i < steps; ++i) {
        std::vector<SimplexVertex> old(simplex);
        LOG(Logger::DBG, simplexToString())
        std::size_t worst = minInSimplex();
        std::size_t best = maxInSimplex();
        SimplexVertex normalReflectionPoint(reflection().second);
        switch (comperator(normalReflectionPoint.evaluate(eval))) {
            case -1 :{  // step 4 in Algo
                LOG(Logger::DBG, "case -1");
                SimplexVertex casePoint(expansion());
                if(casePoint.evaluate(eval) > simplex[best].getFitness()){
                    LOG(Logger::DBG, "double expansion ");
                    simplex[worst] = casePoint;
                }else{
                    LOG(Logger::DBG, "single expansion ");
                    simplex[worst] = normalReflectionPoint;
                }
                break;
            }
            case 1  :{  // step 5,7 in Algo
                LOG(Logger::DBG, "case 1");
                SimplexVertex casePoint(partialOutsideContraction());
                if (casePoint.evaluate(eval) >= normalReflectionPoint.getFitness()) {
                    LOG(Logger::DBG, "patial outside contraction")
                    simplex[worst] = casePoint;
                } else {
                    LOG(Logger::DBG, "total contraction")
                    totalContraction();
                    evalSimplex(eval);
                }
                break;
            }
            case 0  :{  // step 6 in Algo
                LOG(Logger::DBG, "case 0 ");
                SimplexVertex casePoint(partialInsideContraction());
                casePoint.evaluate(eval);
                if (casePoint.getFitness() >= simplex[worst].getFitness()) {
                    LOG(Logger::DBG, "patrial inside contraction is set" << std::endl
                        << casePoint.toString()<< std::endl)
                    simplex[worst] = casePoint;
                }
                break;
            }
            default :{
                SimplexVertex casePoint = reflection().first;
                casePoint.evaluate(eval);
                if (casePoint.getFitness() > simplex[worst].getFitness()) {
                    simplex[worst]= casePoint;
                }
            }
        }
        // step 10
        if (epsilon > 0) {
            if (checkConvergence()) {
                LOG(Logger::DBG, "checkConvergence succes! ")
                initSimplex(simplex[maxInSimplex()]);
                evalSimplex(eval);
                if (stepMultiplicator >= 2) {
                    stepMultiplicator = stepMultiplicator * 0.5;
                } else {
                    break;
                }
            }
        } else {
            if (eq(old, simplex)) {
                if (stepMultiplicator >= 2) {
                    stepMultiplicator = stepMultiplicator * 0.5;
                    initSimplex(simplex[maxInSimplex()]);
                    evalSimplex(eval);
                } else {
                    break;
                }
            }
        }

        fitness = simplex[maxInSimplex()].getFitness();

    }
    LOG(Logger::DBG, "Done steps: " << i);

    fitness = simplex[maxInSimplex()].getFitness();
    return simplex[maxInSimplex()];
}

bool SimplexOptimizer::eq(std::vector<SimplexVertex> simplex1, std::vector<SimplexVertex> simplex2)
{
    for (std::size_t i = 0; i < simplex1.size(); ++i) {
        for (std::size_t j = 0; j < simplex1[i].size(); ++j) {
            if (simplex1[i][j].getValue() != simplex2[i][j].getValue()) {
                return false;
            }
        }
    }
    return true;

}

void SimplexOptimizer::evalSimplex(Evaluator& eval)
{
    for (std::size_t i = 0; i < simplex.size(); ++i) {
        if (simplex[i].getFitness() < 0) {
            simplex[i].evaluate(eval);
        }
    }
}

void SimplexOptimizer::initSimplex(SimulationParameters newParams)
{
    SimplexVertex tmp(newParams);
    simplex = std::vector<SimplexVertex>();
    simplex.push_back(tmp);
    for (std::size_t i = 0; i < tmp.size(); ++i) {
        SimplexVertex tmp2(tmp);
        tmp2[i].setValue(tmp[i].getValue() + stepMultiplicator * stepsizes[i]);

        // if init is called on a border, inverse the direction
        if (tmp2[i].getValue() == tmp[i].getValue()) {
            tmp2[i].setValue(tmp[i].getValue() - stepMultiplicator * stepsizes[i]);
        }

        tmp2.resetFitness();
        simplex.push_back(tmp2);
    }
}

std::size_t SimplexOptimizer::minInSimplex()
{
    std::size_t retval = 0;
    double min = Limits<double>::getMax();

    for (std::size_t i = 0; i < simplex.size(); ++i) {
        if (min >= simplex[i].getFitness()) {
            min = simplex[i].getFitness();
            retval = i;
        }
    }
    return retval;
}

std::size_t SimplexOptimizer::maxInSimplex()
{
    std::size_t retval = 0;
    double max = Limits<double>::getMin();

    for (std::size_t i = 0; i < simplex.size(); ++i) {
        if (max <= simplex[i].getFitness()) {
            max = simplex[i].getFitness();
            retval = i;
        }
    }
    return retval;
}

// returns T1 = overline{x}, T2 = x' from algorithm in the paper
std::pair<SimplexOptimizer::SimplexVertex, SimplexOptimizer::SimplexVertex> SimplexOptimizer::reflection()
{
    std::size_t worst = minInSimplex();
    SimplexVertex t1(simplex[0]);
    SimplexVertex t2(simplex[0]);
    t1.resetFitness();
    t2.resetFitness();
    for (std::size_t j = 0; j < simplex[0].size(); ++j) {
        double tmp=0.0;
        for (std::size_t i = 0; i < simplex.size(); ++i) {
            if (i != worst) {
                tmp += simplex[i][j].getValue();
            }
        }
        tmp = tmp / (simplex.size()-1);
        t1[j].setValue(tmp);
        tmp = 2 * tmp - simplex[worst][j].getValue();
        t2[j].setValue(tmp);
    }
    return std::pair<SimplexVertex, SimplexVertex>(t1, t2);

}

SimplexOptimizer::SimplexVertex SimplexOptimizer::expansion()
{
    std::pair<SimplexVertex, SimplexVertex> reflRes = reflection();
    SimplexVertex retval = simplex[0];
    retval.resetFitness();
    // to use the overloaded operator is not possible here, about over/underflows
    for (std::size_t i = 0; i < simplex[0].size(); ++i) {
        retval[i].setValue(
            reflRes.second[i].getValue()*2 - reflRes.first[i].getValue());
    }
    return retval;
}

SimplexOptimizer::SimplexVertex SimplexOptimizer::partialOutsideContraction()
{
    std::pair<SimplexVertex, SimplexVertex> reflRes = reflection();
    return merge(reflRes.first, reflRes.second);
}

SimplexOptimizer::SimplexVertex SimplexOptimizer::partialInsideContraction()
{
    std::pair<SimplexVertex, SimplexVertex> reflRes = reflection();
    return merge(reflRes.first, simplex[minInSimplex()]);
}

void SimplexOptimizer::totalContraction()
{
    SimplexVertex best = simplex[maxInSimplex()];
    for (std::size_t i = 0; i < simplex.size(); ++i) {
        simplex[i] = merge(best, simplex[i]);
    }
}

bool SimplexOptimizer::checkConvergence()
{
//#define ALTERN_CONVERGENCE_CRITERION
#ifdef ALTERN_CONVERGENCE_CRITERION
    double a= 0.0;
    double b= 0.0;
    for (std::size_t i = 0; i < simplex.size(); ++i) {
        a += simplex[i].getFitness()*simplex[i].getFitness();
        b += simplex[i].getFitness();
    }
    b = b*b*((double)1/(double)(simplex.size()));
    double tmp = (((double) 1 / (double) (simplex.size() - 1)) * (a - b));
#else
    double f_ = 0.0;
    double n = simplex.size()-1;
    for (std::size_t i = 0; i < simplex.size(); ++i) {
        f_ += simplex[i].getFitness();
    }
    f_ *= ((double) 1 / (n + 1.0));
    double tmp = 0.0;
    for (std::size_t i = 0; i < simplex.size(); ++i) {
        tmp += (simplex[i].getFitness() - f_)*(simplex[i].getFitness() - f_);
    }
    tmp *= ((double) 1 / (n + 1.0));
#endif

    LOG(Logger::DBG, "Convergencecheck: " << tmp << " epsilon^2: " << (epsilon * epsilon));

    if (tmp < epsilon * epsilon){
        return true;
    }

    return false;

}

int SimplexOptimizer::comperator(double curFitness)
{
    int retval = -1;

    for (std::size_t i = 0; i < simplex.size(); ++i) {
        if ((simplex[i].getFitness() == curFitness) && (retval == -1)) {
            ++retval;
        }
        if (simplex[i].getFitness() > curFitness) {
            ++retval;
        }
    }
    return retval;
}

SimplexOptimizer::SimplexVertex SimplexOptimizer::merge(
    const SimplexVertex& a,
    const SimplexVertex& b) const
{
    SimplexOptimizer::SimplexVertex result(a);
    result.resetFitness();

    for (std::size_t i = 0; i < result.size(); ++i) {
        double newValue = (a[i].getValue() + b[i].getValue()) * 0.5;
        result[i].setValue(newValue);
    }

    return result;
}

std::string SimplexOptimizer::simplexToString() const
{
    std::stringstream result;
    result << std::endl;

    for (std::size_t i = 0; i < simplex.size(); ++i) {
        result <<  "Vertex " << i << ": ";
        for (std::size_t j = 0;j < simplex[i].size(); ++j) {
            result << simplex[i][j].getValue() << "; ";
        }
        result << "fitness: " << simplex[i].getFitness();
        result << std::endl;
    }
    return result.str();
}

} // namespace LibGeoDecomp

#ifdef _MSC_BUILD
#pragma warning( disable : 4710 )
#endif
