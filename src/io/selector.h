#ifndef LIBGEODECOMP_IO_SELECTOR_H
#define LIBGEODECOMP_IO_SELECTOR_H

namespace LibGeoDecomp {

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
        memberOffset(LibFlatArray::member_ptr_to_offset()(memberPointer)),
        memberName(memberName)
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

    int offset() const
    {
        return memberOffset;
    }

private:
    char CELL:: *memberPointer;
    std::size_t memberSize;
    int memberOffset;
    std::string memberName;
};

}

#endif
