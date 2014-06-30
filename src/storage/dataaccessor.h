#ifndef LIBGEODECOMP_STORAGE_DATAACCESSOR_H
#define LIBGEODECOMP_STORAGE_DATAACCESSOR_H

#include <boost/algorithm/string/case_conv.hpp>

namespace LibGeoDecomp {

/**
 * Manages access to member variables of cells from generic
 * Writer/Steerer classes. DataAccessor usually serves as a base class
 * for a macro-generated specialization (see DEFINE_DATAACCESSOR).
 */
template<typename CELL_TYPE, typename MEMBER_TYPE>
class DataAccessor
{
public:
    DataAccessor(
        const std::string& variableName,
        const std::string& variableType) :
        myName(variableName),
        myType(boost::algorithm::to_upper_copy(variableType))
    {}

    virtual ~DataAccessor()
    {}

    virtual MEMBER_TYPE get(const CELL_TYPE&) = 0;
    virtual void get(const CELL_TYPE&, void*) = 0;
    virtual void set(CELL_TYPE*, void*) = 0;
    virtual std::size_t memberSize() = 0;

    const std::string& name() const
    {
        return myName;
    }

    const std::string type() const
    {
        return myType;
    }

private:
    std::string myName;
    std::string myType;
};

#define DEFINE_DATAACCESSOR(NAME, CELL, MEMBER_TYPE, MEMBER_NAME)       \
    class NAME : public DataAccessor<CELL, MEMBER_TYPE>                 \
    {                                                                   \
    public:                                                             \
        typedef MEMBER_TYPE MemberType;                                 \
                                                                        \
        NAME() :                                                        \
            DataAccessor<CELL, MEMBER_TYPE>(#MEMBER_NAME, #MEMBER_TYPE) \
        {}                                                              \
                                                                        \
        MEMBER_TYPE get(const CELL& in)                                 \
        {                                                               \
            return in.MEMBER_NAME;                                      \
        }                                                               \
                                                                        \
        void get(const CELL& in, void *out)                             \
        {                                                               \
            *static_cast<MEMBER_TYPE*>(out) = in.MEMBER_NAME;           \
        }                                                               \
                                                                        \
        void set(CELL *cell, void *value)                               \
        {                                                               \
            cell->MEMBER_NAME = *(static_cast<MEMBER_TYPE*>(value));    \
        }                                                               \
                                                                        \
        std::size_t memberSize()                                        \
        {                                                               \
            return sizeof(MEMBER_TYPE);                                 \
        }                                                               \
    };

}

#endif
