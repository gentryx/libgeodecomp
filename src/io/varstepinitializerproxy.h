// vim: noai:ts=4:sw=4:expandtab
#ifndef LIBGEODECOMP_IO_VARSTEPINITIALIZERPROXY_H
#define LIBGEODECOMP_IO_VARSTEPINITIALIZERPROXY_H

#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/config.h>
/**
 * The VarStepInitializerProxy is a class to rig the max Steps.
 * It provide the possibility to change the max steps of an exiting
 * intinializer.
 */
namespace LibGeoDecomp {
template<typename CELL>
class VarStepInitializerProxy: public Initializer<CELL>
{
public:
    //friend class PolymorphicSerialization;
    //friend class BoostSerialization;
    //friend class HPXSerialization;

    //TODO i don't no, is this ok?
    typedef typename Initializer<CELL>::Topology Topology;
    const static int DIM = Topology::DIM;

VarStepInitializerProxy(Initializer<CELL> *proxyObj):
     Initializer<CELL>(),
     proxyObj(proxyObj),
     newMaxSteps(proxyObj->maxSteps())
{}

/**
 * change the maxSteps to a new value
 */
void setMaxSteps(unsigned steps){
    maxSteps = steps;
}
/**
 * This funktion returns the raw Value of steps to do
 */
unsigned getMaxSteps()
{
    return newMaxSteps;
}

virtual void grid(GridBase<CELL,DIM> *target) override
{
    proxyObj->grid(target);
}
virtual Coord<DIM> gridDimensions() const override
{
    return proxyObj->gridDimensions();
}

virtual CoordBox<DIM> gridBox() override
{
    return proxyObj->gridBox();
}

virtual unsigned startStep() const override
{
    return proxyObj->startStep();
}

/**
 * This funktion return the step when the simulation
 * will be finished (startStep + getMaxSteps())
 */
virtual unsigned maxSteps() const override
{
    return proxyObj->startStep() + newMaxSteps;
}

virtual Adjacency getAdjacency() const override
{
    return proxyObj->getAdjacency();
}

private:
    // TODO where is the correct point to delete proxyObj
    // TODO use an smart pointer and make it clonable...
    Initializer<CELL> *proxyObj;
    unsigned newMaxSteps;
}; // VarStepInitializerProxy
} // namespace LibGeoDecomp
#endif // LIBGEODECOMP_IO_VARSTEPINITIALIZERPROXY_H

