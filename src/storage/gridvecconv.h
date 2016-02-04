#ifndef LIBGEODECOMP_STORAGE_GRIDVECCONV_H
#define LIBGEODECOMP_STORAGE_GRIDVECCONV_H

#include <libgeodecomp/config.h>
#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/storage/displacedgrid.h>
#include <libgeodecomp/storage/unstructuredgrid.h>
#include <libgeodecomp/storage/unstructuredsoagrid.h>

#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/serialization/shared_ptr.hpp>
#endif

#ifdef LIBGEODECOMP_WITH_HPX
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/binary_filter.hpp>
#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>
#endif

namespace LibGeoDecomp {

template<typename CELL, typename TOPOLOGY, bool TOPOLOGICALLY_CORRECT>
class SoAGrid;

/**
 * Extract a number of cells from a grid type as specified by a Region -- and vice versa.
 */
class GridVecConv
{
public:
    template<typename GRID_TYPE, typename VECTOR_TYPE, typename REGION_TYPE>
    static void gridToVector(
        const GRID_TYPE& grid,
        VECTOR_TYPE *vec,
        const REGION_TYPE& region)
    {
        typedef typename GRID_TYPE::CellType CellType;
        gridToVector(
            grid, vec, region,
            typename APITraits::SelectSoA<CellType>::Value(),
            typename APITraits::SelectBoostSerialization<CellType>::Value());
    }

    template<typename GRID_TYPE, typename VECTOR_TYPE, typename REGION_TYPE>
    static void vectorToGrid(
        const VECTOR_TYPE& vec,
        GRID_TYPE *grid,
        const REGION_TYPE& region)
    {
        typedef typename GRID_TYPE::CellType CellType;
        vectorToGrid(
            vec, grid, region,
            typename APITraits::SelectSoA<CellType>::Value(),
            typename APITraits::SelectBoostSerialization<CellType>::Value());
    }

    template<typename GRID_TYPE, typename VECTOR_TYPE, typename REGION_TYPE>
    static void vectorToGrid(
        VECTOR_TYPE& vec,
        GRID_TYPE *grid,
        const REGION_TYPE& region)
    {
        typedef typename GRID_TYPE::CellType CellType;
        vectorToGrid(
            vec, grid, region,
            typename APITraits::SelectSoA<CellType>::Value(),
            typename APITraits::SelectBoostSerialization<CellType>::Value());
    }

private:

    template<typename CELL_TYPE, typename TOPOLOGY_TYPE, bool TOPOLOGICALLY_CORRECT, typename REGION_TYPE>
    static void gridToVector(
        const DisplacedGrid<CELL_TYPE, TOPOLOGY_TYPE, TOPOLOGICALLY_CORRECT>& grid,
        std::vector<CELL_TYPE> *vec,
        const REGION_TYPE& region,
        APITraits::FalseType,
        APITraits::FalseType)
    {
        if (vec->size() != region.size()) {
            throw std::logic_error("region doesn't match vector size");
        }

        if(vec->size() == 0) {
            return;
        }

        CELL_TYPE *dest = &(*vec)[0];

        for (typename Region<TOPOLOGY_TYPE::DIM>::StreakIterator i = region.beginStreak();
             i != region.endStreak(); ++i) {
            const CELL_TYPE *start = &(grid[i->origin]);
            std::copy(start, start + i->length(), dest);
            dest += i->length();
        }
    }

    template<typename CELL_TYPE, typename TOPOLOGY_TYPE, bool TOPOLOGICALLY_CORRECT, typename REGION_TYPE>
    static void gridToVector(
        const SoAGrid<CELL_TYPE, TOPOLOGY_TYPE, TOPOLOGICALLY_CORRECT>& grid,
        std::vector<char> *vec,
        const REGION_TYPE& region,
        const APITraits::TrueType&,
        const APITraits::FalseType&)
    {
        std::size_t regionSize =
            region.size() *
            SoAGrid<CELL_TYPE, TOPOLOGY_TYPE, TOPOLOGICALLY_CORRECT>::AGGREGATED_MEMBER_SIZE;

        if (vec->size() != regionSize) {
            throw std::logic_error("region doesn't match raw vector's size");
        }

        if(vec->size() == 0) {
            return;
        }

        grid.saveRegion(&(*vec)[0], region);
    }

#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
    template<typename CELL_TYPE, typename TOPOLOGY_TYPE, bool TOPOLOGICALLY_CORRECT, typename REGION_TYPE>
    static void gridToVector(
        const DisplacedGrid<CELL_TYPE, TOPOLOGY_TYPE, TOPOLOGICALLY_CORRECT>& grid,
        std::vector<char> *vec,
        const REGION_TYPE& region,
        const APITraits::FalseType&,
        const APITraits::TrueType&)
    {
        vec->resize(0);
#ifdef LIBGEODECOMP_WITH_HPX
        int archive_flags = boost::archive::no_header;
        archive_flags |= hpx::serialization::disable_data_chunking;
        hpx::serialization::output_archive archive(*vec, archive_flags);
#else
        typedef boost::iostreams::back_insert_device<std::vector<char> > Device;
        Device sink(*vec);
        boost::iostreams::stream<Device> stream(sink);
        boost::archive::binary_oarchive archive(stream);
#endif

        for (typename REGION_TYPE::Iterator i = region.begin(); i != region.end(); ++i) {
            archive & grid[*i];
        }
    }
#endif

#ifdef LIBGEODECOMP_WITH_CPP14
    template<typename CELL_TYPE, std::size_t MATRICES, typename VALUE_TYPE, int C, int SIGMA, typename REGION_TYPE>
    static void gridToVector(
        const UnstructuredGrid<CELL_TYPE, MATRICES, VALUE_TYPE, C, SIGMA>& grid,
        std::vector<CELL_TYPE> *vec,
        const REGION_TYPE& region,
        APITraits::FalseType,
        APITraits::FalseType)
    {
        if (vec->size() != region.size()) {
            throw std::logic_error("region doesn't match vector size");
        }

        if(vec->size() == 0) {
            return;
        }

        CELL_TYPE *dest = &(*vec)[0];

        for (typename Region<1>::StreakIterator i = region.beginStreak();
             i != region.endStreak(); ++i) {
            const CELL_TYPE *start = &(grid[i->origin]);
            std::copy(start, start + i->length(), dest);
            dest += i->length();
        }
    }

    template<typename CELL_TYPE, std::size_t MATRICES, typename VALUE_TYPE, int C, int SIGMA, typename REGION_TYPE, typename BOOST_SERIALIZATION_TYPE>
    static void gridToVector(
        const UnstructuredSoAGrid<CELL_TYPE, MATRICES, VALUE_TYPE, C, SIGMA>& grid,
        std::vector<char> *vec,
        const REGION_TYPE& region,
        const APITraits::TrueType&,
        const BOOST_SERIALIZATION_TYPE&)
    {
        std::size_t regionSize =
            region.size() *
            UnstructuredSoAGrid<CELL_TYPE, MATRICES, VALUE_TYPE, C, SIGMA>::AGGREGATED_MEMBER_SIZE;

        if (vec->size() != regionSize) {
            throw std::logic_error("region doesn't match raw vector's size");
        }

        if(vec->size() == 0) {
            return;
        }

        grid.saveRegion(&(*vec)[0], region);
    }

#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
    template<typename CELL_TYPE, std::size_t MATRICES, typename VALUE_TYPE, int C, int SIGMA, typename REGION_TYPE>
    static void gridToVector(
        const UnstructuredGrid<CELL_TYPE, MATRICES, VALUE_TYPE, C, SIGMA>& grid,
        std::vector<char> *vec,
        const REGION_TYPE& region,
        const APITraits::FalseType&,
        const APITraits::TrueType&)
    {
        vec->resize(0);
#ifdef LIBGEODECOMP_WITH_HPX
        int archive_flags = boost::archive::no_header;
        archive_flags |= hpx::serialization::disable_data_chunking;
        hpx::serialization::output_archive archive(*vec, archive_flags);
#else
        typedef boost::iostreams::back_insert_device<std::vector<char> > Device;
        Device sink(*vec);
        boost::iostreams::stream<Device> stream(sink);
        boost::archive::binary_oarchive archive(stream);
#endif

        for (typename REGION_TYPE::Iterator i = region.begin(); i != region.end(); ++i) {
            archive & grid[*i];
        }
    }
#endif

#endif

    template<typename CELL_TYPE, typename TOPOLOGY_TYPE, bool TOPOLOGICALLY_CORRECT, typename REGION_TYPE>
    static void vectorToGrid(
        const std::vector<CELL_TYPE>& vec,
        DisplacedGrid<CELL_TYPE, TOPOLOGY_TYPE, TOPOLOGICALLY_CORRECT> *grid,
        const REGION_TYPE& region,
        const APITraits::FalseType&,
        const APITraits::FalseType&)
    {
        if (vec.size() != region.size()) {
            throw std::logic_error("vector doesn't match region's size");
        }

        if(vec.size() == 0) {
            return;
        }

        const CELL_TYPE *source = &vec[0];

        for (typename REGION_TYPE::StreakIterator i = region.beginStreak();
             i != region.endStreak();
             ++i) {
            unsigned length = i->length();
            const CELL_TYPE *end = source + length;
            CELL_TYPE *dest = &((*grid)[i->origin]);
            std::copy(source, end, dest);
            source = end;
        }
    }

    template<typename CELL_TYPE, typename TOPOLOGY_TYPE, typename REGION_TYPE, bool TOPOLOGICALLY_CORRECT>
    static void vectorToGrid(
        const std::vector<char>& vec,
        SoAGrid<CELL_TYPE, TOPOLOGY_TYPE, TOPOLOGICALLY_CORRECT> *grid,
        const REGION_TYPE& region,
        const APITraits::TrueType&,
        const APITraits::FalseType&)
    {
        std::size_t regionSize =
            region.size() *
            SoAGrid<CELL_TYPE, TOPOLOGY_TYPE, TOPOLOGICALLY_CORRECT>::AGGREGATED_MEMBER_SIZE;

        if (vec.size() != regionSize) {
            throw std::logic_error("raw vector doesn't match region's size");
        }

        if(vec.size() == 0) {
            return;
        }

        grid->loadRegion(&vec[0], region);
    }

#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
    template<typename CELL_TYPE, typename TOPOLOGY_TYPE, bool TOPOLOGICALLY_CORRECT, typename REGION_TYPE>
    static void vectorToGrid(
        std::vector<char>& vec,
        DisplacedGrid<CELL_TYPE, TOPOLOGY_TYPE, TOPOLOGICALLY_CORRECT> *grid,
        const REGION_TYPE& region,
        const APITraits::FalseType&,
        const APITraits::TrueType&)
    {
#ifdef LIBGEODECOMP_WITH_HPX
        hpx::serialization::input_archive archive(vec, vec.size());
#else
        typedef boost::iostreams::basic_array_source<char> Device;
        Device source(&vec.front(), vec.size());
        boost::iostreams::stream<Device> stream(source);
        boost::archive::binary_iarchive archive(stream);
#endif

        for (typename REGION_TYPE::Iterator i = region.begin(); i != region.end(); ++i) {
            archive & (*grid)[*i];
        }
    }
#endif

#ifdef LIBGEODECOMP_WITH_CPP14
    template<typename CELL_TYPE, std::size_t MATRICES, typename VALUE_TYPE, int C, int SIGMA, typename REGION_TYPE>
    static void vectorToGrid(
        const std::vector<CELL_TYPE>& vec,
        UnstructuredGrid<CELL_TYPE, MATRICES, VALUE_TYPE, C, SIGMA> *grid,
        const REGION_TYPE& region,
        const APITraits::FalseType&,
        const APITraits::FalseType&)
    {
        if (vec.size() != region.size()) {
            throw std::logic_error("vector doesn't match region's size");
        }

        if(vec.size() == 0) {
            return;
        }

        const CELL_TYPE *source = &vec[0];

        for (typename REGION_TYPE::StreakIterator i = region.beginStreak();
             i != region.endStreak();
             ++i) {
            unsigned length = i->length();
            const CELL_TYPE *end = source + length;
            CELL_TYPE *dest = &((*grid)[i->origin]);
            std::copy(source, end, dest);
            source = end;
        }
    }

    template<typename CELL_TYPE, std::size_t MATRICES, typename VALUE_TYPE, int C, int SIGMA, typename REGION_TYPE, typename BOOST_SERIALIZATION_TYPE>
    static void vectorToGrid(
        const std::vector<char>& vec,
        UnstructuredSoAGrid<CELL_TYPE, MATRICES, VALUE_TYPE, C, SIGMA> *grid,
        const REGION_TYPE& region,
        const APITraits::TrueType&,
        const BOOST_SERIALIZATION_TYPE&)
    {
        std::size_t regionSize =
            region.size() *
            UnstructuredSoAGrid<CELL_TYPE, MATRICES, VALUE_TYPE, C, SIGMA>::AGGREGATED_MEMBER_SIZE;

        if (vec.size() != regionSize) {
            throw std::logic_error("raw vector doesn't match region's size");
        }

        if(vec.size() == 0) {
            return;
        }

        grid->loadRegion(&vec[0], region);
    }
#endif
};

}

#endif
