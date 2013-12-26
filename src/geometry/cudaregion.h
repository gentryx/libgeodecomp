#ifndef LIBGEODECOMP_GEOMETRY_CUDAREGION_H
#define LIBGEODECOMP_GEOMETRY_CUDAREGION_H

#include <libgeodecomp/geometry/region.h>
#include <libgeodecomp/storage/cudaarray.h>

#include <cuda.h>
#include <vector>

namespace LibGeoDecomp {

/**
 * Region is optimized for CPUs, CUDARegion is optimized for iteration
 * on GPUs. It does not implement the full Region functionality to
 * avoid redundancy.
 */
template<int DIM>
class CUDARegion
{
public:
    template<typename REGION_TYPE>
    CUDARegion(const REGION_TYPE& region)
    {
        std::vector<int> coordBuffers[DIM];

        for (int i = 0; i < DIM; ++i) {
            coords[i] = CUDAArray<int>(region.size());
            coordBuffers[i].reserve(region.size());
        }

        for (typename REGION_TYPE::Iterator i = region.begin(); i != region.end(); ++i) {
            addCoord(coordBuffers, *i);
        }
    }

    void getCoordPointers(int *pointers) const
    {
        for (int i = 0; i < DIM; ++i) {
            pointers[i] = coords[i].data();
        }
    }

private:
    CUDAArray<int> coords[DIM];

    void addCoord(std::vector<int> coordBuffers[1], const Coord<1>& c) const
    {
        coordBuffers[0].push_back(c[0]);
    }

    void addCoord(std::vector<int> coordBuffers[2], const Coord<2>& c) const
    {
        coordBuffers[0].push_back(c[0]);
        coordBuffers[1].push_back(c[1]);
    }

    void addCoord(std::vector<int> coordBuffers[3], const Coord<3>& c) const
    {
        coordBuffers[0].push_back(c[0]);
        coordBuffers[1].push_back(c[1]);
        coordBuffers[2].push_back(c[2]);
    }
};

}

#endif
