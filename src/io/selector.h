#ifndef LIBGEODECOMP_IO_SELECTOR_H
#define LIBGEODECOMP_IO_SELECTOR_H

#include <libgeodecomp/misc/apitraits.h>
#include <libflatarray/flat_array.hpp>
#include <typeinfo>

namespace LibGeoDecomp {

class APITraits;

namespace SelectorHelpers {

template<typename CELL, typename MEMBER>
class GetMemberOffset
{
public:
    int operator()(MEMBER CELL:: *memberPointer, APITraits::TrueType)
    {
        return LibFlatArray::member_ptr_to_offset()(memberPointer);
    }

    int operator()(MEMBER CELL:: *memberPointer, APITraits::FalseType)
    {
        return -1;
    }
};

/**
 * Primitive datatypes don't have member pointers (or members in the
 * first place). So we provide this primitive implementation to copy
 * them en block.
 */
template<typename CELL>
class PrimitiveSelector
{
public:
    template<typename MEMBER>
    void operator()(const CELL *source, MEMBER *target, const std::size_t length) const
    {
        std::copy(source, source + length, target);
    }

    const std::string& name() const
    {
        return "primitiveType";
    }

    std::size_t sizeOf() const
    {
        return sizeof(CELL);
    }

    int offset() const
    {
        return 0;
    }

    void copyMemberIn(const char *source, CELL *target, int num) const
    {
        const CELL *actualSource = reinterpret_cast<const CELL*>(source);
        std::copy(actualSource, actualSource + num, target);
    }

    void copyMemberOut(const CELL *source, char *target, int num) const
    {
        CELL *actualTarget = reinterpret_cast<CELL*>(target);
        std::copy(source, source + num, actualTarget);
    }

    // intentionally leaving our copyStreak() as it should only be
    // called by SoAGrid, which isn't defined for primitive types
    // anyway.
};

}

/**
 * A Selector can be used by library code to extract data from user
 * code, e.g. so that writers can access a cell's member variable.
 *
 * The code is based on a rather obscure C++ feature. Thanks to Evan
 * Wallace for pointing this out:
 * http://madebyevan.com/obscure-cpp-features/#pointer-to-member-operators
 *
 * fixme: use this class in BOVWriter, RemoteSteerer, VisItWriter, Plotter...
 */
template<typename CELL>
class Selector
{
public:
    template<typename MEMBER>
    Selector(MEMBER CELL:: *memberPointer, std::string memberName) :
        memberPointer(reinterpret_cast<char CELL::*>(memberPointer)),
        memberSize(sizeof(MEMBER)),
        memberTypeIDHandler(&Selector<CELL>::template memberTypeIDImplementation<MEMBER>),
        memberOffset(typename SelectorHelpers::GetMemberOffset<CELL, MEMBER>()(
                         memberPointer,
                         typename APITraits::SelectSoA<CELL>::Value())),
        memberName(memberName),
        copyMemberInHandler(&Selector<CELL>::template copyMemberInImplementation<MEMBER>),
        copyMemberOutHandler(&Selector<CELL>::template copyMemberOutImplementation<MEMBER>),
        copyStreakHandler(&Selector<CELL>::template copyStreakImplementation<MEMBER>)
    {}

    inline char CELL:: *operator*() const
    {
        return memberPointer;
    }

    template<typename MEMBER>
    void operator()(const CELL *source, MEMBER *target, const std::size_t length) const
    {
        MEMBER CELL:: *actualMember = reinterpret_cast<MEMBER CELL::*>(memberPointer);
        for (std::size_t i = 0; i < length; ++i) {
            target[i] = source[i].*actualMember;
        }
    }

    const std::string& name() const
    {
        return memberName;
    }

    std::size_t sizeOf() const
    {
        return memberSize;
    }

    template<typename MEMBER>
    bool checkTypeID() const
    {
        return (*memberTypeIDHandler)(typeid(MEMBER));
    }

    int offset() const
    {
        return memberOffset;
    }

    void copyMemberIn(const char *source, CELL *target, int num) const
    {
        (*copyMemberInHandler)(source, target, num, memberPointer);
    }

    void copyMemberOut(const CELL *source, char *target, int num) const
    {
        (*copyMemberOutHandler)(source, target, num, memberPointer);
    }

    void copyStreak(const char *first, const char *last, char *target) const
    {
        (*copyStreakHandler)(first, last, target);
    }

private:
    char CELL:: *memberPointer;
    std::size_t memberSize;
    bool (*memberTypeIDHandler)(const std::type_info&);
    int memberOffset;
    std::string memberName;
    void (*copyMemberInHandler)(const char *, CELL *, int num, char CELL:: *memberPointer);
    void (*copyMemberOutHandler)(const CELL *, char *, int num, char CELL:: *memberPointer);
    void (*copyStreakHandler)(const char *, const char *, char *);

    template<typename MEMBER>
    static void copyStreakImplementation(const char *first, const char *last, char *target)
    {
        std::copy(
            reinterpret_cast<const MEMBER*>(first),
            reinterpret_cast<const MEMBER*>(last),
            reinterpret_cast<MEMBER*>(target));
    }

    template<typename MEMBER>
    static void copyMemberInImplementation(
        const char *source, CELL *target, int num, char CELL:: *memberPointer)
    {
        MEMBER CELL:: *actualMember = reinterpret_cast<MEMBER CELL:: *>(memberPointer);
        const MEMBER *actualSource = reinterpret_cast<const MEMBER*>(source);

        for (int i = 0; i < num; ++i) {
            (*target).*actualMember = *actualSource;
            ++target;
            ++actualSource;
        }
    }

    template<typename MEMBER>
    static void copyMemberOutImplementation(
        const CELL *source, char *target, int num, char CELL:: *memberPointer)
    {
        MEMBER CELL:: *actualMember = reinterpret_cast<MEMBER CELL:: *>(memberPointer);
        MEMBER *actualTarget = reinterpret_cast<MEMBER*>(target);

        for (int i = 0; i < num; ++i) {
            *actualTarget = (*source).*actualMember;
            ++actualTarget;
            ++source;
        }
    }

    template<typename MEMBER>
    static bool memberTypeIDImplementation(const std::type_info& otherID)
    {
        return typeid(MEMBER) == otherID;
    }

};

/**
 * We provide these specializations to allow our unit tests to use
 * grids with primitive datatypes.
 */
template<>
class Selector<char> : public SelectorHelpers::PrimitiveSelector<char>
{};

template<>
class Selector<int> : public SelectorHelpers::PrimitiveSelector<int>
{};

template<>
class Selector<unsigned> : public SelectorHelpers::PrimitiveSelector<unsigned>
{};

template<>
class Selector<float> : public SelectorHelpers::PrimitiveSelector<float>
{
};

template<>
class Selector<double> : public SelectorHelpers::PrimitiveSelector<double>
{
};

}

#endif
