#ifndef _libgeodecomp_misc_floatcoord_h_
#define _libgeodecomp_misc_floatcoord_h_

namespace LibGeoDecomp {

/**
 * A real valued coordinate class, which contains an optional ID.
 */
template<int DIM>
class FloatCoordBase
{
public:
    std::string toString() const
    {
        std::stringstream s;
        s << "(";
        for (int i = 0; i < DIM - 1; ++i)
            s << c[i] << ",";
        s << c[DIM - 1] << ")";
        return s.str();
    }

    boost::array<double, DIM> c;
    int id;
};

template<int DIM>
class FloatCoord;

template<>
class FloatCoord<1> : public FloatCoordBase<1>
{
public:
    FloatCoord(const double& x = 0, const int _id = 0) 
    {
        c[0] = x;
        id = _id;
    }
};

template<>
class FloatCoord<2> : public FloatCoordBase<2>
{
public:
    FloatCoord(const double& x = 0, const double& y = 0, const int _id = 0) 
    {
        c[0] = x;
        c[1] = y;
        id = _id;
    }
};

template<>
class FloatCoord<3> : public FloatCoordBase<3>
{
public:
    FloatCoord(const double& x = 0, const double& y = 0, const double& z = 0, const int _id = 0) 
    {
        c[0] = x;
        c[1] = y;
        c[2] = z;
        id = _id;
    }
};

}

// fixme: replace this with a unified type trait "HasToString"?
template<typename _CharT, typename _Traits, int DIMENSIONS>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const LibGeoDecomp::FloatCoord<DIMENSIONS>& coord)
{
    __os << coord.toString();
    return __os;
}

#endif
