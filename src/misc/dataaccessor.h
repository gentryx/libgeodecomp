#ifndef LIBGEODECOMP_MISC_DATAACCESSOR_H
#define LIBGEODECOMP_MISC_DATAACCESSOR_H

#include <boost/algorithm/string/case_conv.hpp>

namespace LibGeoDecomp {

/**
 * Manages access to member variables of cells from generic
 * Writer/Steerer classes. DataAccessor usually serves as a base class
 * for a macro-generated specialization (see DEFINE_DATAACCESSOR).
 */
template<typename CELL_TYPE>
// fixme: unify this with BOVWriter infrastucture
// fixme: move to src/io?
class DataAccessor
{
public:
    virtual void getFunction(const CELL_TYPE&, void*) = 0;
    virtual void setFunction(CELL_TYPE*, void*) = 0;
    virtual size_t memberSize() = 0;

    DataAccessor(
        const std::string& variableName,
        const std::string& variableType) :
        name(variableName),
        type(boost::algorithm::to_upper_copy(variableType))
    {}

    virtual ~DataAccessor()
    {}

    const std::string& getName() const
    {
        return name;
    }

    const std::string getType() const
    {
        return type;
    }

private:
    std::string name;
    std::string type;
};

#define DEFINE_DATAACCESSOR(CELL, MEMBER_TYPE, MEMBER_NAME)             \
    class MEMBER_NAME##DataAccessor : public DataAccessor<CELL>         \
    {                                                                   \
    public:                                                             \
        typedef MEMBER_TYPE MemberType;                                 \
                                                                        \
        MEMBER_NAME##DataAccessor() :                                   \
            DataAccessor<CELL>(#MEMBER_NAME, #MEMBER_TYPE)              \
            {}                                                          \
                                                                        \
        void getFunction(const CELL& in, void *out)                     \
        {                                                               \
            *static_cast<MEMBER_TYPE*>(out) = in.MEMBER_NAME;           \
        }                                                               \
                                                                        \
        void setFunction(CELL *cell, void *value)                       \
        {                                                               \
            cell->MEMBER_NAME = *(static_cast<MEMBER_TYPE*>(value));    \
        }                                                               \
                                                                        \
        size_t memberSize()                                             \
        {                                                               \
            return sizeof(MEMBER_TYPE);                                 \
        }                                                               \
    };

}

#endif
