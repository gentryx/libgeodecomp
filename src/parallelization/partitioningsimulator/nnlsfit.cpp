#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#include <stdexcept>
#include <libgeodecomp/parallelization/partitioningsimulator/nnlsfit.h>
#include <libgeodecomp/parallelization/partitioningsimulator/nnls.h>

namespace LibGeoDecomp {

NNLSFit::NNLSFit(const DataPoints& dataPoints)
{
    _observations = dataPoints.size();
    if (_observations < 1) 
        throw std::invalid_argument("Need at least 1 observation.");

    unsigned size = dataPoints.front().size();
    if (size < 2)
        throw std::invalid_argument("Need at least 1 parameter.");
    _params = size - 1;

    if (_observations < _params)
        throw std::invalid_argument("Not enought DataPoints to fit parameters");

     _X = new double[_observations * _params];
     _y = new double[_observations];
     _c = new double[_params];
     _working_space_w = new double[_params];
     _working_space_zz = new double[_observations];
     _working_space_index = new int[_params];

    refit(dataPoints);
}


NNLSFit::~NNLSFit()
{
    delete _X;
    delete _y;
    delete _c;
    delete _working_space_w;
    delete _working_space_zz;
    delete _working_space_index;
}


void NNLSFit::refit(const DataPoints& dataPoints)
{
    validateShape(dataPoints);

    for (unsigned n = 0; n < _observations; n++) {
        for (unsigned p = 0; p < _params; p++) {
            // data must be entered into _X in column major mode
            _X[_observations * p + n] = dataPoints[n][p];
        }
        _y[n] = dataPoints[n].back();
    }

    double rnorm;
    int mode;

    nnls(
            _X, _observations, _observations, _params, _y, _c, &rnorm, 
            _working_space_w, _working_space_zz, _working_space_index, &mode);
}


double NNLSFit::solve(const DVec& inputs) const
{
    if (inputs.size() != _params)
        throw std::invalid_argument("Input size doesn't match data point size");

    double result = 0;
    for (unsigned i = 0; i < _params; i++)
        result += inputs[i] * _c[i];


    return result;
}


void NNLSFit::validateShape(const DataPoints& dataPoints) 
{
    if (dataPoints.size() != _observations)
        throw std::invalid_argument("number of observations doesn't match.");

    for (DataPoints::const_iterator i = dataPoints.begin(); 
            i != dataPoints.end(); i++) {
        if (((*i).size() - 1) != _params)
            throw std::invalid_argument("parameter count doesn't match.");
    }
}


DVec NNLSFit::coefficients() const
{
    DVec result(_params);
    for (unsigned i = 0; i < _params; i++)
        result[i] = _c[i];
    return result;
}

};
#endif
