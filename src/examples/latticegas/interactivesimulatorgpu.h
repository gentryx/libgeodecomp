#ifndef LIBGEODECOMP_EXAMPLES_LATTICEGAS_INTERACTIVESIMULATORGPU_H
#define LIBGEODECOMP_EXAMPLES_LATTICEGAS_INTERACTIVESIMULATORGPU_H

#include <libgeodecomp/examples/latticegas/interactivesimulator.h>

class InteractiveSimulatorGPU : public InteractiveSimulator
{
    Q_OBJECT
    
public:
    InteractiveSimulatorGPU(QObject *parent);
    virtual ~InteractiveSimulatorGPU();
    virtual void loadStates();
    virtual void renderOutput();
    virtual void update();
};

#endif
