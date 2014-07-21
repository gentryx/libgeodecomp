// vim: noai:ts=4:sw=4:expandtab
#ifndef LIBGEODECOMP_MISC_SIMPLEXOPTIMIZER_H
#define LIBGEODECOMP_MISC_SIMPLEXOPTIMIZER_H


#include <libgeodecomp/misc/optimizer.h>
#include <libgeodecomp/misc/simulationparameters.h>
#include <iostream>
#include <sstream>
#include <cfloat>
#include <libgeodecomp/io/logger.h>
#include <utility>

#define LIBGEODECOMP_DEBUG_LEVEL 4
#define ALTERN_CONVERGENCE_CRITERION

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
        std::string toString()
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
        const SimplexVertex operator+(const SimplexVertex& b) 
        {
            SimplexVertex result(*this);
            if (this->size()==b.size()){
                for (std::size_t i = 0; i < b.size(); ++i) {
                    result[i].setValue(this->operator[](i).getValue() + b[i].getValue());
                }
                result.setFitness(-1);
                return result;
            }
            return b;
        }
        const SimplexVertex operator+(const double& b)
        {
            SimplexVertex result(*this);
            for (std::size_t i = 0; i < this->size(); ++i) {
                result[i].setValue(this->operator[](i).getValue() + b);
            }
            result.setFitness(-1);
            return result;
        }
        const SimplexVertex operator-(const SimplexVertex& b)
        {
            SimplexVertex result(*this);
            for (std::size_t i = 0; i < b.size(); ++i) {
                result[i].setValue(this->operator[](i).getValue() - b[i].getValue());
            }     
            result.setFitness(-1);
            return result;
        }
        const SimplexVertex operator*(const SimplexVertex& b) 
        {
            SimplexVertex result(*this);
            if (this->size()==b.size()){
                for (std::size_t i = 0; i < b.size(); ++i) {
                    result[i].setValue(this->operator[](i).getValue() * b[i].getValue());
                }
                result.setFitness(-1);
                return result;
            }
            return b;
        }
        const SimplexVertex operator*(const double& b)
        {
            SimplexVertex result(*this);
            for (std::size_t i = 0; i < this->size(); ++i) {
                result[i].setValue(this->operator[](i).getValue() * b);
            }
            result.setFitness(-1);
            return result;
        }

    protected:
        void setFitness(double fitness)
        {
            this->fitness = fitness;
        }
    private:
        double fitness;
    }; //SimplexVertex

    explicit SimplexOptimizer(SimulationParameters params);
    virtual SimulationParameters operator()(int steps, Evaluator& eval);

private:
    
    void evalSimplex(Evaluator& eval);
    std::size_t minInSimplex();
    std::size_t maxInSimplex();
    void totalContraction();
    bool checkTermination();
    bool checkConvergence();
    std::pair<SimplexVertex, SimplexVertex> reflection();
    SimplexVertex expansion();
    SimplexVertex partialOutsideContraction();
    SimplexVertex partialInsideContraction();
    bool eq(vector<SimplexVertex> simplex1, vector<SimplexVertex> simplex2);
    std::vector<SimplexVertex> simplex;
    int comperator(double fitness);
    std::string simplexToString();
    double s;   // init Stepsize
    double c;   // 
    double epsilon;
};

// TODO have to bin into SimplexOptimizer.cpp after coding
SimplexOptimizer::SimplexOptimizer(SimulationParameters params) : 
    Optimizer(params),
    s(1),
    c(8),
   epsilon(30)
{
    // n+1 vertices neded
    simplex.push_back(SimplexVertex(params));
    std::cout << "in Construktor!" << simplexToString() << std::endl;
    for (std::size_t i = 0; i < params.size(); ++i) {
        SimplexVertex tmp(params);
        tmp[i].setValue(params[i].getValue() + c * s); 
        simplex.push_back(tmp);
    }
}

SimulationParameters SimplexOptimizer::operator()(int steps, Evaluator& eval)
{
    vector<SimplexVertex> old(simplex);
    evalSimplex(eval);
    for (int i = 0; i < steps && checkTermination(); ++i) {
    
    std::cout<< std::endl << " nach expansion " << std::endl << "simplex[0].getMin(): " << simplex[0][3].getMin() << " simplex[0].getMax: " << simplex[0][3].getMax() << std::endl;
        vector<SimplexVertex> old(simplex);
        LOG(Logger::DBG, simplexToString())
        std::size_t worst = minInSimplex();
        std::size_t best = maxInSimplex();
        SimplexVertex newPoint(reflection().second);
        // TODO it can  crash if a border is crossing
        switch (comperator(newPoint.evaluate(eval))) {
            case -1 :{  // step 4 in Algo
                LOG(Logger::DBG, "case -1");
                SimplexVertex casePoint(expansion());
                if(casePoint.evaluate(eval) > simplex[best].getFitness()){
                    LOG(Logger::DBG, "double expansion ");
                    simplex[worst] = casePoint; 
                }else{
                    LOG(Logger::DBG, "single expansion ");
                    simplex[worst] = newPoint;
                }
                break;
            }
            case 1  :{  // step 5,7 in Algo
                LOG(Logger::DBG, "case 1");
                SimplexVertex casePoint(partialOutsideContraction());
                if (casePoint.evaluate(eval) >= newPoint.getFitness()) {
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
                    LOG(Logger::DBG, "patrial inside contraction is set" << std::endl << casePoint.toString()<< std::endl)
                    simplex[worst]=casePoint;
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
        if (checkConvergence()) {
            LOG(Logger::DBG, "checkConvergence succes! ")
            SimplexVertex tmp = simplex[maxInSimplex()];            
            simplex = vector<SimplexVertex>();
            simplex.push_back(tmp);
            for (std::size_t i = 0; i < tmp.size(); ++i) {
                SimplexVertex tmp2(tmp);
                tmp2[i].setValue(tmp[i].getValue() + c * s); 
                tmp2.evaluate(eval);
                simplex.push_back(tmp2);
            }
            if (eq(old,simplex)) {
                if (c > 1) {
                    c = c * 0.5;
                } else {
                    LOG(Logger::DBG, "succesful search!!")
                    break;
                }
            }

        }
        
        fitness = simplex[maxInSimplex()].getFitness();
        
    }
    return simplex[maxInSimplex()];
}

bool SimplexOptimizer::eq(vector<SimplexVertex> simplex1, vector<SimplexVertex> simplex2)
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
            simplex[i].evaluate(eval);
    }
}

std::size_t SimplexOptimizer::minInSimplex()
{
    std::size_t retval = 0;
    double min = DBL_MAX;
    for (std::size_t i = 0; i < simplex.size(); ++i) {
        if (min > simplex[i].getFitness()) {
            min = simplex[i].getFitness();
            retval = i;
        }
    }
    return retval;
}

std::size_t SimplexOptimizer::maxInSimplex()
{
    std::size_t retval = 0;
    double max = DBL_MIN;
    for (std::size_t i = 0; i < simplex.size(); ++i) {
        if (max < simplex[i].getFitness()) {
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
    SimplexVertex retval(simplex[0]);
    retval = retval * 2.0;
    retval = retval - reflRes.first;
    return retval;
}

SimplexOptimizer::SimplexVertex SimplexOptimizer::partialOutsideContraction() 
{
    std::pair<SimplexVertex, SimplexVertex> reflRes = reflection();
    SimplexVertex retval(simplex[0]);
    retval = reflRes.first + reflRes.second;
    retval = retval * 0.5;
    return retval;
}

SimplexOptimizer::SimplexVertex SimplexOptimizer::partialInsideContraction() 
{
    std::pair<SimplexVertex, SimplexVertex> reflRes = reflection();
    SimplexVertex retval(simplex[0]);
    retval = reflRes.first + simplex[minInSimplex()];
    retval = retval * 0.5;
    return retval;

}

void SimplexOptimizer::totalContraction() 
{
    SimplexVertex best = simplex[maxInSimplex()];
    for (std::size_t i = 0; i < simplex.size(); ++i) {
        SimplexVertex result(simplex[0]);
        result = best + simplex[i];
        result = result *0.5;
        simplex[i] = result;
    }
}

bool SimplexOptimizer::checkConvergence() 
{
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
    LOG(Logger::DBG, "Convergencecheck: " << tmp << " epsilonn^2: " << (epsilon *epsilon))
    if(tmp < epsilon * epsilon){
        return true;
    }
    return false;

}

bool SimplexOptimizer::checkTermination() 
{
    // the mathematical criterium wouldn't work with descreat point!?!
    for (std::size_t i = 0; i < simplex[0].size(); ++i) {
        for (std::size_t j = 1; j < simplex.size(); ++j) {
            if (simplex[0][i].getValue() != simplex[j][i].getValue()) {
                return true;
            }
        }
    }
    return false;
}

// return == -1 => all fitness in vertex are lower as fitness 
// return == 0  one fitness in vertex is equal all others are lower 
// return > 0 Nr of Parameters are lower than fitness
int SimplexOptimizer::comperator(double fitness) 
{
    int retval = -1;
    for (std::size_t i = 0; i < simplex.size(); ++i) {
        if (simplex[i].getFitness() == fitness && retval == -1) {
            ++retval;
        }
        if (simplex[i].getFitness()>fitness) {
            ++retval;
        }
    }
    return retval;
}

std::string SimplexOptimizer::simplexToString() 
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

#endif //LIBGEODECOMP_MISC_SIMPLEXOPTIMIZER_H
