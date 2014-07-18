// vim: noai:ts=4:sw=4:expandtab
#ifndef LIBGEODECOMP_MISC_SIMPLEXOPTIMIZER_H
#define LIBGEODECOMP_MISC_SIMPLEXOPTIMIZER_H


#include <libgeodecomp/misc/optimizer.h>
#include <libgeodecomp/misc/simulationparameters.h>
#include <iostream>
#include <sstream>
#include <cfloat>
#include <libgeodecomp/io/logger.h>

#define LIBGEODECOMP_DEBUG_LEVEL 4
// TODO (nice to have) Simplex can inherit from Simulation Parameters

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
            for(std::size_t j = 0;j < this->size(); ++j){
                result << this->operator[](j).getValue() << "; ";
            }
            result << "fitness: " << getFitness();
            result << std::endl;
            
            return result.str();
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
    SimplexVertex reflection();    
    SimplexVertex expansion();
    SimplexVertex partialOutsideContraction();
    SimplexVertex partialInsideContraction();
    std::vector<SimplexVertex> simplex;
    int comperator(double fitness);
    std::string simplexToString();
    double s;   // init Stepsize
    double c;   // 
};

// TODO have to bin into SimplexOptimizer.cpp after coding
SimplexOptimizer::SimplexOptimizer(SimulationParameters params) : 
    Optimizer(params),
    s(1),
    c(1)
{
    // n+1 vertices neded
    simplex.push_back(SimplexVertex(params));
    for(std::size_t i = 0; i < params.size(); ++i){
        SimplexVertex tmp(params);
        tmp[i].setValue(params[i].getValue() + c * s); 
        simplex.push_back(tmp);
    }
}

SimulationParameters SimplexOptimizer::operator()(int steps, Evaluator& eval)
{
    evalSimplex(eval);
    for(int i = 0; i < steps && checkTermination(); ++i){
        LOG(Logger::DBG, simplexToString())
        std::size_t worst = minInSimplex();
        std::size_t best = maxInSimplex();
        SimplexVertex newPoint(reflection());
        newPoint.evaluate(eval);
        // TODO it can  crash if a border is crossing
        switch(comperator(newPoint.evaluate(eval))){
            case -1 :{  // step 4 in Algo
                LOG(Logger::DBG, "case -1");
                SimplexVertex casePoint(expansion());
                casePoint.evaluate(eval);
                if(casePoint.getFitness() > simplex[best].getFitness()){
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
                casePoint.evaluate(eval);
                if(newPoint.getFitness() < casePoint.getFitness()){
                    simplex[worst] = casePoint;
                }else{
                    totalContraction();
                    evalSimplex(eval);
                    continue;
                }
                break;
            }
            case 0  :{  // step 6 in Algo
                LOG(Logger::DBG, "case 0 ");
                SimplexVertex casePoint(partialInsideContraction());
                casePoint.evaluate(eval);
                if(casePoint.getFitness() >= simplex[worst].getFitness()){
                    simplex[worst]=casePoint;
                }
            }
            default :{
                std::stringstream log;
                log << newPoint.toString();
                log << "default case, comperator value:  "<< comperator(newPoint.getFitness()); 
                if(simplex[worst].getFitness() > newPoint.getFitness()){
                    totalContraction();
                    evalSimplex(eval);

                }else{
                    simplex[worst]=newPoint;
                }
                LOG(Logger::DBG, log.str());

            }
        }
        fitness = simplex[best].getFitness();

    }
    return simplex[0];
}

void SimplexOptimizer::evalSimplex(Evaluator& eval){
    for(std::size_t i = 0; i < simplex.size(); ++i){
        //if(simplex[i].getFitness()<0){ // don't work now
            simplex[i].evaluate(eval);
        //}
    }
}

std::size_t SimplexOptimizer::minInSimplex()
{
    std::size_t retval = 0;
    double min = DBL_MAX;
    for(std::size_t i = 0; i < simplex.size(); ++i){
        if(min > simplex[i].getFitness()){
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
    for(std::size_t i = 0; i < simplex.size(); ++i){
        if(max < simplex[i].getFitness()){
            max = simplex[i].getFitness();
            retval = i;
        }
    }
    return retval;
}

SimplexOptimizer::SimplexVertex SimplexOptimizer::reflection()
{
    std::size_t worst = minInSimplex();
    SimulationParameters retval = simplex[0];
    for(std::size_t j = 0; j<simplex[0].size(); ++j){
        double tmp=0.0;
        for(std::size_t i = 0; i < simplex.size(); ++i){
            if(i != worst){
                tmp += simplex[i][j].getValue();
            }
        }
        tmp = tmp / (simplex[0].size()-1); 
        tmp = 2 * tmp - simplex[worst][j].getValue();
        retval[j].setValue(tmp);
    }
    return SimplexVertex(retval);
}

SimplexOptimizer::SimplexVertex SimplexOptimizer::expansion(){

    std::size_t worst = minInSimplex();
    SimplexVertex retval = simplex[0];
    for(std::size_t j = 0; j<simplex[0].size(); ++j){
        double tmp=0.0;
        double tmp2=0.0;
        for(std::size_t i = 0; i < simplex.size(); ++i){
            if(i != worst){
                tmp += simplex[i][j].getValue();
            }
        }
        tmp = tmp / (simplex[0].size()-1); 
        tmp2 = tmp;
        tmp = 2 * tmp - simplex[worst][j].getValue();
        retval[j].setValue(2 * tmp - tmp2);
    }
    return retval;
}

SimplexOptimizer::SimplexVertex SimplexOptimizer::partialOutsideContraction(){
    
    std::size_t worst = minInSimplex();
    SimplexVertex retval = simplex[0];
    for(std::size_t j = 0; j<simplex[0].size(); ++j){
        double tmp=0.0;
        double tmp2=0.0;
        for(std::size_t i = 0; i < simplex.size(); ++i){
            if(i != worst){
                tmp += simplex[i][j].getValue();
            }
        }
        tmp = tmp / (simplex[0].size()-1); 
        tmp2 = tmp; // xBar
        tmp = 2 * tmp - simplex[worst][j].getValue();   //x'
        retval[j].setValue(0.5  * (tmp + tmp2));
    }
    return retval;
    
}
SimplexOptimizer::SimplexVertex SimplexOptimizer::partialInsideContraction(){
    std::size_t worst = minInSimplex();
    SimplexVertex retval = simplex[0];
    for(std::size_t j = 0; j<simplex[0].size(); ++j){
        double tmp=0.0;
        for(std::size_t i = 0; i < simplex.size(); ++i){
            if(i != worst){
                tmp += simplex[i][j].getValue();
            }
        }
        tmp = tmp / (simplex[0].size()-1); 
        tmp = 2 * tmp - simplex[worst][j].getValue();
        retval[j].setValue(2 * (tmp + simplex[worst][j].getValue()));
    }
    return retval;
}

void SimplexOptimizer::totalContraction(){
    SimplexVertex best = simplex[maxInSimplex()];
    for(std::size_t i = 0; i < simplex.size(); ++i){
        for(std::size_t j = 0; j < simplex[i].size(); ++j){  
            simplex[i][j].setValue(
                0.5 * (best[j].getValue()
                +simplex[i][j].getValue()));
        }
    }
}

bool SimplexOptimizer::checkTermination(){
    // the mathematical criterium wouldn't work with descreat point!?!
    for(std::size_t i = 0; i < simplex[0].size(); ++i){
        for(std::size_t j = 1; j < simplex.size(); ++j){
            if(simplex[0][i].getValue() != simplex[j][i].getValue()){
                return true;
            }
        }
    }
    return false;
}

// return == -1 => all fitness in vertex are lower as fitness 
// return == 0  one fitness in vertex is equal all others are lower 
// return > 0 Nr of Parameters are lower than fitness
int SimplexOptimizer::comperator(double fitness){
    int retval = -1;
    for(std::size_t i = 0; i < simplex.size(); ++i){
        if(simplex[i].getFitness()==fitness && retval == -1){
            ++retval;
        }
        if(simplex[i].getFitness()>fitness){
            ++retval;
        }
    }
    return retval;
}

std::string SimplexOptimizer::simplexToString(){
    std::stringstream result;
    result << std::endl;
    for(std::size_t i = 0; i < simplex.size(); ++i){
        result <<  "Vertex " << i << ": ";
        for(std::size_t j = 0;j < simplex[i].size(); ++j){
            result << simplex[i][j].getValue() << "; ";
        }
        result << "fitness: " << simplex[i].getFitness();
        result << std::endl;
    }
    return result.str();
}

} // namespace LibGeoDecomp

#endif //LIBGEODECOMP_MISC_SIMPLEXOPTIMIZER_H
