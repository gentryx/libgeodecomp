#ifndef _libgeodecomp_misc_coordbox_h_
#define _libgeodecomp_misc_coordbox_h_

#include <iostream>
#include <stdexcept>
#include <libgeodecomp/misc/coord.h>
#include <libgeodecomp/misc/topologies.h>

namespace LibGeoDecomp {

template<int DIM>
class CoordBoxSequence;

template<int DIM>
class CoordBox 
{
    friend class Typemaps;

public:    

    Coord<DIM> origin;
    Coord<DIM> dimensions;

    explicit CoordBox(const Coord<DIM>& origin_ = Coord<DIM>(), 
                      const Coord<DIM>& dimensions_ = Coord<DIM>()) :
        origin(origin_),
        dimensions(dimensions_)
    {}

    inline bool operator==(const CoordBox& other) const
    {
        return (origin == other.origin &&
                dimensions == other.dimensions);
    }


    inline bool operator!=(const CoordBox& other) const
    {
        return ! (*this == other);
    }


    inline bool inBounds(const Coord<DIM>& coord) const
    {
        Coord<DIM> relativeCoord = coord - origin;
        return !Topologies::IsOutOfBoundsHelper<
            DIM - 1, Coord<DIM>, typename Topologies::Cube<DIM>::Topology>()(
                relativeCoord, dimensions);
    }


    std::string toString() const
    {
        std::ostringstream temp;
        temp << "origin: " << origin << "\n"
             << "dimensions: " << dimensions << "\n";
        return temp.str();
    }

    /**
     * @return a box that has been symmetrically enlarged/shrunk on all
     * sides by @a rimWidth. A box with non-positive width or height will
     * be set to a box with size 0 and origin 0^DIM.
     */
    // fixme: remove
    CoordBox resized(const int& rimWidth) const
    {
        Coord<DIM> offset = CoordDiagonal<DIM>()(rimWidth);
        Coord<DIM> newOrigin = origin - offset;
        Coord<DIM> newDimensions = dimensions + offset * 2;

        Coord<DIM> test = CoordDiagonal<DIM>()(1).max(newDimensions);
        if (test != newDimensions)
            return CoordBox();


        return CoordBox(newOrigin, newDimensions);
    }


    // fixme: remove
    bool intersects(const CoordBox& other) const
    {
        Coord<DIM> maxOrigin(
            std::max(origin.x(), other.origin.x()),
            std::max(origin.y(), other.origin.y()));
        return inBounds(maxOrigin) && other.inBounds(maxOrigin);
    }

    // fixme: remove
    CoordBox intersect(const CoordBox& other) const
    {
        Coord<DIM> opposite1 = this->originOpposite();
        Coord<DIM> opposite2 = other.originOpposite();

        Coord<DIM> resUL = this->origin.max(other.origin);
        Coord<DIM> resLR = opposite1.min(opposite2);
        resLR += CoordDiagonal<DIM>()(1);

        Coord<DIM> zero;
        Coord<DIM> dim = zero.max(resLR - resUL);

        return CoordBox(resUL, dim);
    }

    // fixme: remove
    bool contains(const CoordBox& other) const
    {
        return intersect(other) == other;
    }

    // fixme: remove
    inline Coord<DIM> originOpposite() const 
    { 
        Coord<DIM> ret = origin + dimensions + CoordDiagonal<DIM>()(-1);

        return ret;
    }

    inline unsigned size() const 
    { 
        return dimensions.prod();
    }

    inline CoordBoxSequence<DIM> sequence() const
    {
        return CoordBoxSequence<DIM>(*this);
    }
};

// fixme: replace this poor class by a proper iterator
template<int DIM>
class CoordBoxSequence
{
public: 
    CoordBoxSequence(const CoordBox<DIM>& nbox) :
        box(nbox),
        index(0)
    {}

    inline virtual ~CoordBoxSequence() {}

    inline bool hasNext() const
    { 
        return index < box.size();
    }

    inline Coord<DIM> next() 
    {
        if (!hasNext()) 
            throw std::out_of_range(
                "next() called at end of CoordBoxSequence");            
        
        Coord<DIM> offset = IndexToCoord<DIM>()(index, box.dimensions);
        index += 1;

        return box.origin + offset;
    }

private:
    CoordBox<DIM> box;
    unsigned index;
};

/**
 * The MPI typemap generator need to find out for which template
 * parameter values it should generate typemaps. It does so by
 * scanning all class members. Therefore this dummy class forces the
 * typemap generator to create MPI datatypes for CoordBoxs with the
 * dimensions as specified below.
 */
class CoordBoxMPIDatatypeHelper
{
    friend class Typemaps;
    CoordBox<1> a;
    CoordBox<2> b;
    CoordBox<3> c;
};

}


template<typename _CharT, typename _Traits, int _Dimensions>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const LibGeoDecomp::CoordBox<_Dimensions>& rect)
{
    __os << rect.toString();
    return __os;
}

#endif
