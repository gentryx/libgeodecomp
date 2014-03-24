#ifndef LIBGEODECOMP_IO_SELECTOR_H
#define LIBGEODECOMP_IO_SELECTOR_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/io/logger.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libflatarray/flat_array.hpp>
#include <typeinfo>

#ifdef LIBGEODECOMP_WITH_SILO
#include <silo.h>
#endif

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

#ifdef LIBGEODECOMP_WITH_SILO
template<typename MEMBER>
class GetSiloTypeID
{
public:
    inline int operator()()
    {
        LOG(WARN, "Warning: using type unknown to Silo for output");
        return DB_NOTYPE;
    }
};

template<>
class GetSiloTypeID<int>
{
public:
    inline int operator()()
    {
        return DB_INT;
    }
};

template<>
class GetSiloTypeID<short int>
{
public:
    inline int operator()()
    {
        return DB_SHORT;
    }
};

template<>
class GetSiloTypeID<float>
{
public:
    inline int operator()()
    {
        return DB_FLOAT;
    }
};

template<>
class GetSiloTypeID<double>
{
public:
    inline int operator()()
    {
        return DB_DOUBLE;
    }
};

template<>
class GetSiloTypeID<char>
{
public:
    inline int operator()()
    {
        return DB_CHAR;
    }
};

template<>
class GetSiloTypeID<long long>
{
public:
    inline int operator()()
    {
        return DB_LONG_LONG;
    }
};

#endif

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

    const std::string name() const
    {
        return "primitiveType";
    }

    std::size_t sizeOfMember() const
    {
        return sizeof(CELL);
    }

    std::size_t sizeOfExternal() const
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
 *
 * fixme: extend this to make use of custom filters via function pointers:
 * - c-tor should save an extra function pointer copyStreakAoSHandler()
 * - call copyStreakAoSHandler() in operator()
 *   - default copyStreakAoSHandler() implementation is current implementation
 *   - alternate implementation can be given in c-tor via function pointer instead of member pointer
 *   - note that result type (ATM "MEMBER") can differ
 * - rename copyStreakHandler to copyStreakSoAHandler
 *   - specify source attribute of cell via member pointer
 */
template<typename CELL>
class Selector
{
public:
    /**
     * Base class for adding user-defined data filters to a selector.
     * This can be used to do on-the-fly data extraction, scale
     * conversion for live output etc. without having to rewrite a
     * complete ParallelWriter output plugin.
     *
     * It is suggested to derive from Filter instead of FilterBase, as
     * the latter has some convenience functionality already in place.
     */
    class FilterBase
    {
    public:
        virtual ~FilterBase()
        {}

        virtual std::size_t sizeOf() const = 0;
#ifdef LIBGEODECOMP_WITH_SILO
        virtual int siloTypeID() const = 0;
#endif
        virtual void copyStreakIn(const char *first, const char *last, char *target) = 0;
        virtual void copyStreakOut(const char *first, const char *last, char *target) = 0;
        virtual void copyMemberIn(
            const char *source, CELL *target, int num, char CELL:: *memberPointer) = 0;
        virtual void copyMemberOut(
            const CELL *source, char *target, int num, char CELL:: *memberPointer) = 0;
        virtual bool checkExternalTypeID(const std::type_info& otherID) const = 0;
    };

    /**
     * Derive from this class if you wish to add custom data
     * adapters/converters to your Selector.
     */
    template<typename MEMBER, typename EXTERNAL>
    class Filter : public FilterBase
    {
    public:
        std::size_t sizeOf() const
        {
            return sizeof(EXTERNAL);
        }

#ifdef LIBGEODECOMP_WITH_SILO
        int siloTypeID() const
        {
            return SelectorHelpers::GetSiloTypeID<EXTERNAL>()();
        }
#endif

        virtual void copyStreakInImpl(const EXTERNAL *first, const EXTERNAL *last, MEMBER *target) {}
        virtual void copyStreakOutImpl(const MEMBER *first, const MEMBER *last, EXTERNAL *target) {}
        virtual void copyMemberInImpl(
            const EXTERNAL *source, CELL *target, int num, MEMBER CELL:: *memberPointer) {}

        virtual void copyMemberOutImpl(
            const CELL *source, EXTERNAL *target, int num, MEMBER CELL:: *memberPointer) {}

        void copyStreakIn(const char *first, const char *last, char *target)
        {
            copyStreakInImpl(
                reinterpret_cast<const EXTERNAL*>(first),
                reinterpret_cast<const EXTERNAL*>(last),
                reinterpret_cast<MEMBER*>(target));
        }

        void copyStreakOut(const char *first, const char *last, char *target)
        {
            copyStreakOutImpl(
                reinterpret_cast<const MEMBER*>(first),
                reinterpret_cast<const MEMBER*>(last),
                reinterpret_cast<EXTERNAL*>(target));
        }

        void copyMemberIn(
            const char *source, CELL *target, int num, char CELL:: *memberPointer)
        {
            copyMemberInImpl(
                reinterpret_cast<const EXTERNAL*>(source),
                target,
                num,
                reinterpret_cast<MEMBER CELL:: *>(memberPointer));
        }

        void copyMemberOut(
            const CELL *source, char *target, int num, char CELL:: *memberPointer)
        {
            copyMemberOutImpl(
                source,
                reinterpret_cast<EXTERNAL*>(target),
                num,
                reinterpret_cast<MEMBER CELL:: *>(memberPointer));
        }

        bool checkExternalTypeID(const std::type_info& otherID) const
        {
            return typeid(EXTERNAL) == otherID;
        }
    };

    template<typename MEMBER, typename EXTERNAL>
    class DefaultFilter : public Filter<MEMBER, EXTERNAL>
    {
    public:
        virtual void copyStreakInImpl(const EXTERNAL *first, const EXTERNAL *last, MEMBER *target)
        {
            std::copy(first, last, target);
        }

        virtual void copyStreakOutImpl(const MEMBER *first, const MEMBER *last, EXTERNAL *target)
        {
            std::copy(first, last, target);
        }

        virtual void copyMemberInImpl(
            const EXTERNAL *source, CELL *target, int num, MEMBER CELL:: *memberPointer)
        {
            for (int i = 0; i < num; ++i) {
                target[i].*memberPointer = source[i];
            }
        }

        virtual void copyMemberOutImpl(
            const CELL *source, EXTERNAL *target, int num, MEMBER CELL:: *memberPointer)
        {
            for (int i = 0; i < num; ++i) {
                target[i] = source[i].*memberPointer;
            }
        }
    };

    template<typename MEMBER>
    Selector(MEMBER CELL:: *memberPointer, const std::string& memberName) :
        memberPointer(reinterpret_cast<char CELL::*>(memberPointer)),
        memberSize(sizeof(MEMBER)),
        externalSize(sizeof(MEMBER)),
        memberOffset(typename SelectorHelpers::GetMemberOffset<CELL, MEMBER>()(
                         memberPointer,
                         typename APITraits::SelectSoA<CELL>::Value())),
        memberName(memberName),
#ifdef LIBGEODECOMP_WITH_SILO
        memberSiloTypeID(SelectorHelpers::GetSiloTypeID<MEMBER>()()),
#endif
        filter(new DefaultFilter<MEMBER, MEMBER>)
    {}

    template<typename MEMBER>
    Selector(MEMBER CELL:: *memberPointer, const std::string& memberName, const boost::shared_ptr<FilterBase>& filter) :
        memberPointer(reinterpret_cast<char CELL::*>(memberPointer)),
        memberSize(sizeof(MEMBER)),
        externalSize(filter->sizeOf()),
        memberOffset(typename SelectorHelpers::GetMemberOffset<CELL, MEMBER>()(
                         memberPointer,
                         typename APITraits::SelectSoA<CELL>::Value())),
        memberName(memberName),
#ifdef LIBGEODECOMP_WITH_SILO
        memberSiloTypeID(filter->siloTypeID()),
#endif
        filter(filter)
    {}

    inline const std::string& name() const
    {
        return memberName;
    }

    inline std::size_t sizeOfMember() const
    {
        return memberSize;
    }

    inline std::size_t sizeOfExternal() const
    {
        return externalSize;
    }

    // fixme: use this in public member functions
    template<typename MEMBER>
    inline bool checkTypeID() const
    {
        return filter->checkExternalTypeID(typeid(MEMBER));
    }

    /**
     * The member's offset in LibFlatArray's SoA memory layout
     */
    inline int offset() const
    {
        return memberOffset;
    }

    /**
     * Read the data from source and set the corresponding member of
     * each CELL at target. Only useful for AoS memory layout.
     */
    inline void copyMemberIn(const char *source, CELL *target, int num) const
    {
        filter->copyMemberIn(source, target, num, memberPointer);
    }

    /**
     * Read the member of all CELLs at source and store them
     * contiguously at target. Only useful for AoS memory layout.
     */
    inline void copyMemberOut(const CELL *source, char *target, int num) const
    {
        filter->copyMemberOut(source, target, num, memberPointer);
    }

    /**
     * This is a helper function for writing members of a SoA memory
     * layout.
     */
    inline void copyStreakIn(const char *first, const char *last, char *target) const
    {
        filter->copyStreakIn(first, last, target);
    }

    /**
     * This is a helper function for reading members from a SoA memory
     * layout.
     */
    inline void copyStreakOut(const char *first, const char *last, char *target) const
    {
        filter->copyStreakOut(first, last, target);
    }

#ifdef LIBGEODECOMP_WITH_SILO
    int siloTypeID() const
    {
        return filter->siloTypeID();
    }
#endif

private:
    char CELL:: *memberPointer;
    std::size_t memberSize;
    std::size_t externalSize;
    int memberOffset;
    std::string memberName;
    int memberSiloTypeID;
    boost::shared_ptr<FilterBase> filter;
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
