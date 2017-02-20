#ifndef LIBGEODECOMP_IO_CLONABLEINITIALIZER_H
#define LIBGEODECOMP_IO_CLONABLEINITIALIZER_H

#include <libgeodecomp/io/initializer.h>

namespace LibGeoDecomp {

/**
 * This extension of the Initializer interface allows library classes
 * to create copies (clones) of a particular instance, which is
 * necessary in some use cases, e.g. simulation factories or the HPX
 * backend.
 *
 * We're adding this as a separate class to avoid forcing all users to
 * add a clone() to their Initializers (e.g. via inheritance from
 * Clonable<>. We can still add the default cause (copy c-tor) via a
 * wrapper class.
 */
template<typename CELL>
class ClonableInitializer : public Initializer<CELL>
{
public:
    typedef void IsClonable;

    /**
     * "virtual copy constructor". This function may be called
     * whenever a deep copy of an Initializer is needed instead of a
     * plain pointer copy.
     *
     * Advice to implementers: use CRTP (
     * http://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
     * ) to implement this automagically -- see other Initializer
     * implementations for advice on this subject.
     */
    virtual ClonableInitializer<CELL> *clone() const = 0;
};

}

#endif
