#ifndef LIBGEODECOMP_IO_CLONABLEINITIALIZERWRAPPER_H
#define LIBGEODECOMP_IO_CLONABLEINITIALIZERWRAPPER_H

#include <libgeodecomp/io/clonableinitializer.h>
#include <libgeodecomp/misc/apitraits.h>

namespace LibGeoDecomp {

template<typename INITIALIZER> class ClonableInitializerWrapper;

namespace ClonableInitializerWrapperHelpers
{

template<typename INITIALIZER, typename IS_CLONABLE=void>
class Wrap
{
public:
    typedef typename INITIALIZER::Cell Cell;

    ClonableInitializer<Cell> *operator()(const INITIALIZER& initializer)
    {
        return new ClonableInitializerWrapper<INITIALIZER>(initializer);
    }
};

template<typename INITIALIZER>
class Wrap<INITIALIZER, typename INITIALIZER::IsClonable>
{
public:
    typedef typename INITIALIZER::Cell Cell;

    ClonableInitializer<Cell> *operator()(const INITIALIZER& initializer)
    {
        return initializer.clone();
    }
};

}

/**
 * This class adds a default clone() to any Initializer via its copy
 * c-tor. This is primarily a convenience for users who are this way
 * not forced to inherit from Clonable.
 */
template<class INITIALIZER>
class ClonableInitializerWrapper : public ClonableInitializer<typename INITIALIZER::Cell>
{
public:
    typedef typename INITIALIZER::Cell Cell;
    typedef typename INITIALIZER::Topology Topology;
    const static int DIM = Topology::DIM;

    static ClonableInitializer<Cell> *wrap(const INITIALIZER& initializer)
    {
        return ClonableInitializerWrapperHelpers::Wrap<INITIALIZER>()(initializer);
    }

    ClonableInitializerWrapper(const INITIALIZER& initializer) :
        delegate(initializer)
    {}

    virtual void grid(GridBase<Cell, DIM> *target)
    {
        delegate.grid(target);
    }

    virtual CoordBox<DIM> gridBox()
    {
        return delegate.gridBox();
    }

    virtual Coord<DIM> gridDimensions() const
    {
        return delegate.gridDimensions();
    }

    virtual unsigned startStep() const
    {
        return delegate.startStep();
    }

    virtual unsigned maxSteps() const
    {
        return delegate.maxSteps();
    }

    virtual ClonableInitializer<Cell> *clone() const
    {
        return new ClonableInitializerWrapper<INITIALIZER>(INITIALIZER(delegate));
    }

private:
    INITIALIZER delegate;
};

}

#endif
