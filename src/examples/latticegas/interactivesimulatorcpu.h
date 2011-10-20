#ifndef _libgeodecomp_examples_latticegas_interactivesimulatorcpu_h_
#define _libgeodecomp_examples_latticegas_interactivesimulatorcpu_h_

#include <libgeodecomp/examples/latticegas/interactivesimulator.h>

class InteractiveSimulatorCPU : public InteractiveSimulator
{
    Q_OBJECT
    
public:
    InteractiveSimulatorCPU(QObject *parent);
    virtual ~InteractiveSimulatorCPU();
    virtual void loadStates();
    virtual void renderOutput();
    virtual void update();

private:
    std::vector<BigCell> gridOld;
    std::vector<BigCell> gridNew;
};

#endif
