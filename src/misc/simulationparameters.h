#ifndef LIBGEODECOMP_MISC_SIMULATIONPARAMETERS_H
#define LIBGEODECOMP_MISC_SIMULATIONPARAMETERS_H

// HPX' config needs to be included before Boost's config:
#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX
#include <hpx/config.hpp>
#endif

#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <boost/shared_ptr.hpp>
#include <stdexcept>

namespace LibGeoDecomp {

namespace SimulationParametersHelpers {

/**
 * Virtual interface which allows the implementation of auto-tuners
 * and parameter optimizers without them having to unterstand the
 * actual meaning of the parameters. For that all parameters are
 * mapped to an interval [min, max[ in R.
 *
 * See the unit tests for an explanation on how to use this interface.
 */
class OptimizableParameter
{
public:
    virtual ~OptimizableParameter()
    {}

    /**
     * returns the lower bound of the interval. The lower bound is
     * included in the interval.
     */
    virtual double getMin() const = 0;

    /**
     * returns the upper bound of the interval. The upper bound is
     * excluded from the interval.
     */
    virtual double getMax() const = 0;

    /**
     * The granularity gives the minimum value that a parameter needs
     * to change in order to actually affect the model.
     *
     * Most parameters won't actually have a real-valued valuation.
     * For these the granularity is almost always 1.
     */
    virtual double getGranularity() const = 0;

    /**
     * Returns a real-valued representation of the parameter's current
     * value.
     */
    virtual double getValue() const = 0;

    /**
     * Sets the parameter, based on the given real value. Note that
     * rounding and truncation based on the granularity may occur.
     */
    virtual void setValue(double newValue) = 0;

    /**
     * Move the parameter by the offset given by step. Step sizes
     * below granularity may have no effect.
     */
    virtual void operator+=(double step) = 0;

    /**
     * Pretty-printed string repesentation of parameter (domain and
     * value), most useful for debug output
     */
    virtual std::string toString() const = 0;
};

/**
 * Base class which is implemented by TypedParameter.
 */
class Parameter : public OptimizableParameter
{
public:
    virtual ~Parameter()
    {}

    virtual Parameter *clone() const = 0;

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

    double sanitizeIndex(double index) const
    {
        if (index < getMin()) {
            index = getMin();
        }
        if (index >= getMax()) {
            index = getMax() - getGranularity();
        }

        return index;
    }
};

/**
 * A TypedParameter allows retrieval of the stored value. This can be
 * used by user code (e.g. a simulation factory) to utilize the
 * parameters determined by an Optimizer.
 */
template<typename VALUE_TYPE>
class TypedParameter : public Parameter
{
public:
    // These are required to silence nvcc's warnings. We intend to
    // both, override and overload these operators. More explanations
    // here: http://stackoverflow.com/questions/12871711/virtual-function-override-intended-error
    using Parameter::operator=;
    using Parameter::operator==;

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

/**
 * An interval is a parameter that can take on any value withing range
 * a certain minimum (included) and maximum (excluded) value:
 * [minimum, maximum[
 */
template<typename VALUE_TYPE>
class Interval : public TypedParameter<VALUE_TYPE>
{
public:
    using TypedParameter<VALUE_TYPE>::current;
    using Parameter::sanitizeIndex;

    Interval(const VALUE_TYPE minimum, const VALUE_TYPE maximum, const double granularity = 1) :
        TypedParameter<VALUE_TYPE>(minimum),
        minimum(minimum),
        maximum(maximum),
        granularity(granularity),
        index(0)
    {}

    Parameter *clone() const
    {
        return new Interval<VALUE_TYPE>(*this);
    }

    double getMin() const
    {
        return 0;
    }

    double getMax() const
    {
        return maximum - minimum;
    }

    double getValue() const
    {
        return index;
    }

    void setValue(double newValue)
    {
        if (granularity > 0.0) {
           index = newValue;
           long nrOfSteps = index / granularity;
           index = nrOfSteps * granularity;
           index = sanitizeIndex(index);
        } else {
            index = sanitizeIndex(newValue);
        }
        current = minimum + index;
    }

    double getGranularity() const
    {
        return granularity;
    }

    void operator+=(double step)
    {
        if (granularity > 0.0) {
            index = index + step;
            long nrOfSteps = index / granularity;
            index = nrOfSteps * granularity;
            index = sanitizeIndex(index);
        } else {
            index += step;
            index = sanitizeIndex(index);
        }
        current = minimum + index;
    }

    std::string toString() const
    {
        std::stringstream buf;
        buf << "Interval([" << minimum << ", " << maximum << "], " << index << ")";
        return buf.str();
    }

private:
    VALUE_TYPE minimum;
    VALUE_TYPE maximum;
    double granularity;
    VALUE_TYPE index;
};

/**
 * Such a parameter may take on any value contained in the vector.
 * Their values are encoded via their relative index within the
 * vector.
 */
template<typename VALUE_TYPE>
class DiscreteSet : public TypedParameter<VALUE_TYPE>
{
public:
    using TypedParameter<VALUE_TYPE>::current;
    using Parameter::sanitizeIndex;

    explicit DiscreteSet(const std::vector<VALUE_TYPE>& elements) :
        TypedParameter<VALUE_TYPE>(elements.front()),
        elements(elements),
        index(0)
    {}

    Parameter *clone() const
    {
        return new DiscreteSet<VALUE_TYPE>(*this);
    }

    double getMin() const
    {
        return 0;
    }

    double getMax() const
    {
        return elements.size();
    }

    double getValue() const
    {
        return index;
    }

    void setValue(double newValue)
    {
        index = sanitizeIndex(index);
        current = elements[index];
    }

    double getGranularity() const
    {
        return 1;
    }

    void operator+=(double step)
    {
        index += step;
        index = sanitizeIndex(index);

        current = elements[index];
    }

    std::string toString() const
    {
        std::stringstream buf;
        buf << "DiscreteSet(" << elements << ", " << index << ")";
        return buf.str();
    }

private:
    std::vector<VALUE_TYPE> elements;
    int index;
};

}

/**
 * Encapsulates a set of parameters (ranges and values) which form the
 * input for an Optimizer and which will in turn forward them to an
 * objective function (goal function).
 */
class SimulationParameters
{
public:
    typedef boost::shared_ptr<SimulationParametersHelpers::Parameter> ParamPointerType;

    SimulationParameters()
    {}

    SimulationParameters(const SimulationParameters& other) :
        names(other.names)
    {
        for (std::size_t i = 0; i < other.size(); ++i) {
            parameters.push_back(ParamPointerType(other[i].clone()));
        }
    }

    template<typename VALUE_TYPE>
    void addParameter(const std::string& name, const VALUE_TYPE& minimum, const VALUE_TYPE& maximum, const double granularity = 1.0)
    {
        names[name] = parameters.size();
        parameters.push_back(
            ParamPointerType(
                new SimulationParametersHelpers::Interval<VALUE_TYPE>(minimum, maximum, granularity)));
    }

    template<typename VALUE_TYPE>
    void addParameter(const std::string& name, const std::vector<VALUE_TYPE>& elements)
    {
        names[name] = parameters.size();
        parameters.push_back(
            ParamPointerType(
                new SimulationParametersHelpers::DiscreteSet<VALUE_TYPE>(elements)));
    }

    template<typename VALUE_TYPE>
    void replaceParameter(const std::string& name, const VALUE_TYPE& minimum, const VALUE_TYPE& maximum, const double granularity = 1.0)
    {
        parameters[names[name]] = ParamPointerType(
            new SimulationParametersHelpers::Interval<VALUE_TYPE>(minimum, maximum, granularity));
    }

    template<typename VALUE_TYPE>
    void replaceParameter(const std::string& name, const std::vector<VALUE_TYPE>& elements)
    {
        parameters[names[name]] = ParamPointerType(
            new SimulationParametersHelpers::DiscreteSet<VALUE_TYPE>(elements));
    }

    SimulationParametersHelpers::Parameter& operator[](const std::string& name)
    {
        return *parameters[names[name]];
    }

    const SimulationParametersHelpers::Parameter& operator[](const std::string& name) const
    {
        if (names.find(name) != names.end()){
            return *parameters[names.find(name)->second];
        } else {
            throw std::invalid_argument("SimulationParameters[\"name\"] get invalid name!");
        }
    }

    SimulationParametersHelpers::Parameter& operator[](std::size_t index)
    {
        return *parameters[index];
    }

    const SimulationParametersHelpers::Parameter& operator[](std::size_t index) const
    {
        return *parameters[index];
    }

    std::string toString() const
    {
        std::stringstream buf;
        buf << "SimulationParameters(\n";
        for (std::map<std::string, int>::const_iterator i = names.begin(); i != names.end(); ++i) {
            buf << "  " << i->first << " => " << parameters[i->second]->toString() << "\n";
        }
        buf << ")\n";

        return buf.str();
    }

    std::size_t size() const
    {
        return parameters.size();
    }

protected:
    std::map<std::string, int> names;
    std::vector<ParamPointerType> parameters;
};

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const SimulationParameters& params)
{
    __os << params.toString();
    return __os;
}

}

#endif
