#ifndef LIBGEODECOMP_IO_CLONABLEINITIALIZERWRAPPER_H
#define LIBGEODECOMP_IO_CLONABLEINITIALIZERWRAPPER_H

#include <libgeodecomp/io/clonableinitializer.h>
#include <libgeodecomp/misc/apitraits.h>

namespace LibGeoDecomp {

template<typename INITIALIZER> class ClonableInitializerWrapper;

namespace ClonableInitializerWrapperHelpers {

/**
 * Switch which add clone() to an Initializer by wrapping it...
 */
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

/**
 * ...but leaves ClonableInitializers untouched.
 */
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

    /**
     * These c-tors forward arguments to an c-tor of INITIALIZER. Up
     * to 8 arguments are accepted. 8 constructor arguments shoult be
     * enough for everybody(tm).
     */
    template<typename T1>
    static ClonableInitializer<Cell> *wrap(const T1& arg1)
    {
        return ClonableInitializerWrapperHelpers::Wrap<INITIALIZER>()(
            INITIALIZER(arg1));
    }

    template<typename T1, typename T2>
    static ClonableInitializer<Cell> *wrap(const T1& arg1, const T2& arg2)
    {
        return ClonableInitializerWrapperHelpers::Wrap<INITIALIZER>()(
            INITIALIZER(arg1, arg2));
    }

    template<typename T1, typename T2, typename T3>
    static ClonableInitializer<Cell> *wrap(const T1& arg1, const T2& arg2, const T3& arg3)
    {
        return ClonableInitializerWrapperHelpers::Wrap<INITIALIZER>()(
            INITIALIZER(arg1, arg2, arg3));
    }

    template<typename T1, typename T2, typename T3, typename T4>
    static ClonableInitializer<Cell> *wrap(const T1& arg1, const T2& arg2, const T3& arg3, const T4& arg4)
    {
        return ClonableInitializerWrapperHelpers::Wrap<INITIALIZER>()(
            INITIALIZER(arg1, arg2, arg3, arg4));
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5>
    static ClonableInitializer<Cell> *wrap(const T1& arg1, const T2& arg2, const T3& arg3, const T4& arg4, const T5& arg5)
    {
        return ClonableInitializerWrapperHelpers::Wrap<INITIALIZER>()(
            INITIALIZER(arg1, arg2, arg3, arg4, arg5));
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
    static ClonableInitializer<Cell> *wrap(const T1& arg1, const T2& arg2, const T3& arg3, const T4& arg4, const T5& arg5, const T6& arg6)
    {
        return ClonableInitializerWrapperHelpers::Wrap<INITIALIZER>()(
            INITIALIZER(arg1, arg2, arg3, arg4, arg5, arg6));
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
    static ClonableInitializer<Cell> *wrap(const T1& arg1, const T2& arg2, const T3& arg3, const T4& arg4, const T5& arg5, const T6& arg6, const T7& arg7)
    {
        return ClonableInitializerWrapperHelpers::Wrap<INITIALIZER>()(
            INITIALIZER(arg1, arg2, arg3, arg4, arg5, arg6, arg7));
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
    static ClonableInitializer<Cell> *wrap(const T1& arg1, const T2& arg2, const T3& arg3, const T4& arg4, const T5& arg5, const T6& arg6, const T7& arg7, const T8& arg8)
    {
        return ClonableInitializerWrapperHelpers::Wrap<INITIALIZER>()(
            INITIALIZER(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8));
    }

    explicit ClonableInitializerWrapper(const INITIALIZER& initializer) :
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
