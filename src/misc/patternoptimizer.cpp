// vim: noai:ts=4:sw=4:expandtab
#include <libgeodecomp/misc/patternoptimizer.h>
#include <libgeodecomp/io/logger.h>
#include <cmath>
#include <iostream>
#include <sstream>

namespace LibGeoDecomp{

PatternOptimizer::PatternOptimizer(SimulationParameters params):
    Optimizer(params),
    stepFaktor(2),
    maxSteps(8)
{
    for(std::size_t i = 0; i < params.size(); ++i) {
        double dimsize = Optimizer::params[i].getMax()
            - Optimizer::params[i].getMin();
        stepwidth.push_back(dimsize/stepFaktor);
        double tmp = stepwidth[i]/std::pow(2,maxSteps);
        // minStepwidht >= granularity
        // it will be used as abort criterion
        if(tmp < params[i].getGranularity())
            tmp=params[i].getGranularity();
        minStepwidth.push_back(tmp);
    }
}

// returns false if all parameters alrady on minimum
bool PatternOptimizer::reduceStepwidth()
{
    bool allWasMin =true;
    std::stringstream log;   
    log << "Reduce Stepwidth:" << std::endl;
    for(size_t i = 0;i < stepwidth.size(); ++i){
        log << "Dimension "<< i << ": " << stepwidth[i];
        if(stepwidth[i] <= minStepwidth[i]){
            log << " --> " << stepwidth[i] << "; " << std::endl;
            stepwidth[i] = minStepwidth[i];
            continue;
        }
        stepwidth[i]=stepwidth[i] / 2;
        if(stepwidth[i]<= minStepwidth[i]){
            stepwidth[i] = minStepwidth[i];
        }
        allWasMin = false;
        log << " --> " << stepwidth[i] << "; " << std::endl;
    }
    LOG(Logger::DBG,log.str())
    return !allWasMin;
}

std::vector<SimulationParameters> PatternOptimizer::genPattern(SimulationParameters middle)
    {
        std::vector<SimulationParameters> result(middle.size() * 2 + 1);
        result[0]=middle;
        for(std::size_t i = 0; i < middle.size(); ++i){
            SimulationParameters tmp1(middle),tmp2(middle);
            tmp1[i] += stepwidth[i];
            tmp2[i] += (stepwidth[i] * -1.0);
            result[1+i*2] = tmp1;
            result[2+i*2] = tmp2;
        }	
        return result;
    }

std::size_t PatternOptimizer::getMaxPos(std::vector<SimulationParameters> pattern, Evaluator& eval, std::size_t oldMiddle)
{	
    std::size_t retval = 0;
    double newFitness;
    double oldFitness = Optimizer::fitness;
    for(std::size_t i = 1; i < pattern.size(); ++i){ //i = 1 middle don't need to be evaluate again
        // all pattern[i] with the same coordinates as middle, oldMiddle don't need to be evaluate
        if(pattern[0][(i-1)/2].getValue() == pattern[i][(i-1)/2].getValue() || i ==oldMiddle){
            if(oldMiddle == i && oldFitness > Optimizer::fitness){
                retval = i;
            }
            continue;
        }
        newFitness = eval(pattern[i]);
        if(newFitness>Optimizer::fitness){
            retval = i;
            Optimizer::fitness = newFitness;
        }
    }
    return retval;
}

SimulationParameters PatternOptimizer::operator()(int steps,Evaluator & eval)
{
    SimulationParameters middle = Optimizer::params;
    std::vector<SimulationParameters> pattern = genPattern(middle);
    std::size_t maxPos = 0;
    for (int k = 0; k < steps; ++k){
        pattern = genPattern(middle);
        LOG(Logger::DBG, patternToString(pattern))
        std::size_t maxPos = getMaxPos(pattern, eval,maxPos);
        if(maxPos == 0){            // center was the Maximum
            if(!reduceStepwidth()){  // abort test
                LOG(Logger::DBG,  "fitness: " << Optimizer::fitness)
                return middle;
            }
        }else{                      // do next step
            middle = pattern[maxPos];
        }
    }
    //unreachable
    return middle;
}

std::string PatternOptimizer::patternToString(std::vector<SimulationParameters> pattern){
    std::stringstream result;
    result << "Pattern: " << std::endl;
    for(std::size_t i = 0; i<pattern.size(); ++i){
        if(i==0){
            result << "Middle:       ";
        }else{
            result << "Direction: " << i << " :	";
        }
        for(std::size_t j = 0; j < pattern[i].size(); ++j){
            result << pattern[i][j].getValue() << " - ";
        }
        result << std::endl;
    }
    result << std::endl;
    return result.str();
}
} //namespace LibGeoDecomp
