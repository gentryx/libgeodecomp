#ifndef _libgeodecomp_misc_dataaccessor_h_
#define _libgeodecomp_misc_dataaccessor_h_

#include <boost/algorithm/string/case_conv.hpp>

namespace LibGeoDecomp
{

/**
 * contains information about cell variables
 *
 * generated in simulation
 */
template<typename CELL_TYPE>
class DataAccessor
{
public:
    virtual void getFunction(const CELL_TYPE&, void*) = 0;
    virtual void setFunction(CELL_TYPE*, void*) = 0;

    DataAccessor(std::string variableName, std::string variableType)
    {
        name = variableName;
        type = boost::algorithm::to_upper_copy(variableType);
    }

    std::string getName()
    {
        return name;
    }

    std::string getType()
    {
        return type;
    }
private:
    std::string name;
    std::string type;
};

#define DEFINE_DATAACCESSOR(CELL, MEMBER_TYPE, MEMBER_NAME)                 \
class MEMBER_NAME##DataAccessor                                             \
        : public DataAccessor<CELL>                                         \
{                                                                           \
public:                                                                     \
    void getFunction(const CELL& in, void *out){                            \
        *static_cast<MEMBER_TYPE*>(out) = in.MEMBER_NAME;                   \
    }                                                                       \
    void setFunction(CELL* cell, void *value){                              \
        cell->MEMBER_NAME = *(static_cast<MEMBER_TYPE*>(value));            \
    }                                                                       \
    MEMBER_NAME##DataAccessor() :                                           \
            DataAccessor<CELL>                                              \
                    (#MEMBER_NAME, #MEMBER_TYPE) {}                         \
};

}

#endif
