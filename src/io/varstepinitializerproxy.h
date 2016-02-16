// vim: noai:ts=4:sw=4:expandtab
#ifndef LIBGEODECOMP_IO_VARSTEPINITIALIZERPROXY_H
#define LIBGEODECOMP_IO_VARSTEPINITIALIZERPROXY_H

#include <memory>
#include <libgeodecomp/io/initializer.h>
#include <libgeodecomp/config.h>
#include <libgeodecomp/io/clonableinitializer.h>

#ifdef LIBGEODECOMP_WITH_CPP14

/**
 * The VarStepInitializerProxy is a class to rig the max Steps.
 * It provide the possibility to change the max steps of an exiting
 * intinializer.
 */
namespace LibGeoDecomp {

template<typename CELL>
class VarStepInitializerProxy : public ClonableInitializer<CELL>
{
public:
    typedef typename Initializer<CELL>::Topology Topology;
    const static int DIM = Topology::DIM;

    VarStepInitializerProxy(Initializer<CELL> *proxyObj):
        ClonableInitializer<CELL>(),
        proxyObj(boost::shared_ptr<Initializer<CELL> >(proxyObj)),
        newMaxSteps(proxyObj->maxSteps())
    {}

    /**
     * change the maxSteps to a new value
     */
    void setMaxSteps(unsigned steps) {
        newMaxSteps = steps;
    }

    /**
     * This function returns the remaining steps to be simulated
     */
    unsigned getMaxSteps() const
    {
        return newMaxSteps;
    }

    /**
     * This function returns a shared_ptr to the original Initializer
     */
    boost::shared_ptr<Initializer<CELL> > getInitializer()
    {
        return proxyObj;
    }

    //------------------- inherited functions from Initializer ------------------
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
     * This function return the step when the simulation
     * will be finished (startStep + getMaxSteps())
     */
    virtual unsigned maxSteps() const override
    {
        return proxyObj->startStep() + newMaxSteps;
    }

    virtual boost::shared_ptr<Adjacency> getAdjacency() const override
    {
        return proxyObj->getAdjacency();
    }

    //--------------- inherited functions from Clonableinitializer --------------
    virtual ClonableInitializer<CELL> *clone() const override
    {
        return new VarStepInitializerProxy<CELL>(*this);
    }

private:
    VarStepInitializerProxy(VarStepInitializerProxy<CELL>* o) :
        proxyObj(o->proxyObj),
        newMaxSteps(o->newMaxSteps)
    {}

    boost::shared_ptr<Initializer<CELL> > proxyObj;
    unsigned newMaxSteps;
};

}

#endif

#endif

