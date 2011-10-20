#ifndef _libgeodecomp_examples_latticegas_interactivesimulatorgpu_h_
#define _libgeodecomp_examples_latticegas_interactivesimulatorgpu_h_

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

private:
    // fixme: get rid of these
    std::vector<BigCell> gridOld;
    std::vector<BigCell> gridNew;
};

#endif
