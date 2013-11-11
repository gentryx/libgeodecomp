#ifndef LIBGEODECOMP_GEOMETRY_FLOATCOORDBASE_H
#define LIBGEODECOMP_GEOMETRY_FLOATCOORDBASE_H

namespace LibGeoDecomp {

template<int DIM>
class FloatCoordBase
{
    friend class Typemaps;
    friend class Serialization;
public:

    virtual ~FloatCoordBase() {}

    inline
    std::string toString() const
    {
        std::stringstream s;
        s << "(";
        for (int i = 0; i < DIM - 1; ++i)
            s << c[i] << ", ";
        s << c[DIM - 1] << ")";
        return s.str();
    }

    inline double& operator[](const int i)
    {
        return c[i];
    }

    inline const double& operator[](const int i) const
    {
        return c[i];
    }

    double c[DIM];
};

class FloatCoordBaseMPIDatatypeHelper
{
    friend class Typemaps;
    FloatCoordBase<1> a;
    FloatCoordBase<2> b;
    FloatCoordBase<3> c;
};

}

#endif
