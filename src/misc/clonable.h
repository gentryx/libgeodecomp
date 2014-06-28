#ifndef LIBGEODECOMP_MISC_CLONABLE_H
#define LIBGEODECOMP_MISC_CLONABLE_H

#include <stdexcept>

namespace LibGeoDecomp {

/**
 * This class adds a virtual copy constructor to child classes. It's
 * implemented using CRTP.
 */
template<typename BASE, typename IMPLEMENTATION>
class Clonable : public BASE
{
public:
    friend class Serialization;

    /**
     * these c-tors simply delegate to the BASE type, which includes
     * the default c-tor, sans parameters.
     */
    explicit
    Clonable()
    {}

    template<typename T1>
    explicit
    Clonable(const T1& p1) :
        BASE(p1)
    {}

    template<typename T1, typename T2>
    explicit
    Clonable(const T1& p1, const T2& p2) :
        BASE(p1, p2)
    {}

    template<typename T1, typename T2, typename T3>
    explicit
    Clonable(const T1& p1, const T2& p2, const T3& p3) :
        BASE(p1, p2, p3)
    {}

    BASE *clone() const
    {
        const IMPLEMENTATION *ret = dynamic_cast<const IMPLEMENTATION*>(this);

        if (ret == 0) {
            throw std::logic_error("Wrong IMPLEMENTATION type chosen for Clonable");
        }

        return new IMPLEMENTATION(*ret);
    }
};

}

#endif
