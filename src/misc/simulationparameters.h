#ifndef LIBGEODECOMP_MISC_SIMULATIONPARAMETERS_H
#define LIBGEODECOMP_MISC_SIMULATIONPARAMETERS_H

// HPX' config needs to be included before Boost's config:
#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX
#include <hpx/config.hpp>
#endif

#include <boost/shared_ptr.hpp>
#include <stdexcept>

namespace LibGeoDecomp {

namespace SimulationParametersHelpers {

class Parameter
{
public:
    virtual ~Parameter()
    {}

    virtual operator std::string() const
    {
        throw std::logic_error("illegal cast to std::string");
    }

    virtual operator bool() const
    {
        throw std::logic_error("illegal cast to bool");
    }

    virtual operator int() const
    {
        throw std::logic_error("illegal cast to int");
    }

    virtual operator double() const
    {
        throw std::logic_error("illegal cast to double");
    }


    virtual void operator=(const std::string& other)
    {
        throw std::logic_error("illegal assignment from string");
    }

    virtual void operator=(const char *other)
    {
        *this = std::string(other);
    }

    virtual void operator=(const bool& other)
    {
        throw std::logic_error("illegal assignment from bool");
    }

    virtual void operator=(const int& other)
    {
        throw std::logic_error("illegal assignment from int");
    }

    virtual void operator=(const double& other)
    {
        throw std::logic_error("illegal assignment from double");
    }

    virtual bool operator==(const std::string& other) const
    {
        return false;
    }

    virtual bool operator==(const char *other) const
    {
        return *this == std::string(other);
    }

    virtual bool operator==(const bool& other) const
    {
        return false;
    }

    virtual bool operator==(const int& other) const
    {
        return false;
    }

    virtual bool operator==(const double& other) const
    {
        return false;
    }

    template<typename OTHER_TYPE>
    bool operator!=(const OTHER_TYPE& other) const
    {
        return !(*this == other);
    }
};

template<typename VALUE_TYPE>
class TypedParameter : public Parameter
{
public:
    explicit TypedParameter(const VALUE_TYPE& current) :
        current(current)
    {}

    virtual operator VALUE_TYPE() const
    {
        return current;
    }

    virtual void operator=(const VALUE_TYPE& other)
    {
        current = other;
    }

    virtual bool operator==(const VALUE_TYPE& other) const
    {
        return current == other;
    }

protected:
    VALUE_TYPE current;
};

template<typename VALUE_TYPE>
class Interval : public TypedParameter<VALUE_TYPE>
{
public:
    Interval(const VALUE_TYPE minimum, const VALUE_TYPE maximum) :
        TypedParameter<VALUE_TYPE>(minimum),
        minimum(minimum),
        maximum(maximum)
    {}

private:
    VALUE_TYPE minimum;
    VALUE_TYPE maximum;
};

template<typename VALUE_TYPE>
class DiscreteSet : public TypedParameter<VALUE_TYPE>
{
public:
    explicit DiscreteSet(const std::vector<VALUE_TYPE>& elements) :
        TypedParameter<VALUE_TYPE>(elements.front()),
        elements(elements)
    {}

private:
    std::vector<VALUE_TYPE> elements;
};

}

class SimulationParameters
{
public:
    template<typename VALUE_TYPE>
    void addParameter(const std::string& name, const VALUE_TYPE& minimum, const VALUE_TYPE& maximum)
    {
        parameters[name].reset(new SimulationParametersHelpers::Interval<VALUE_TYPE>(minimum, maximum));
    }

    template<typename VALUE_TYPE>
    void addParameter(const std::string& name, const std::vector<VALUE_TYPE>& elements)
    {
        parameters[name].reset(new SimulationParametersHelpers::DiscreteSet<VALUE_TYPE>(elements));
    }

    SimulationParametersHelpers::Parameter& operator[](const std::string& name)
    {
        return *parameters[name];
    }

private:
    std::map<std::string, boost::shared_ptr<SimulationParametersHelpers::Parameter> > parameters;
};

}

#endif
