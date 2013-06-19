#ifndef LIBGEODECOMP_EXAMPLES_LATTICEGAS_INTERACTIVESIMULATORCPU_H
#define LIBGEODECOMP_EXAMPLES_LATTICEGAS_INTERACTIVESIMULATORCPU_H

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
    std::vector<unsigned> frame;
};

#endif
