#ifndef LIBGEODECOMP_GEOMETRY_CUDAREGION_H
#define LIBGEODECOMP_GEOMETRY_CUDAREGION_H

#include <libgeodecomp/geometry/region.h>
#include <libflatarray/cuda_array.hpp>

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
        int size = DIM * region.size();
        coords = LibFlatArray::cuda_array<int>(size);
        std::vector<int> coordsBuffer(size);
        std::size_t index = 0;

        for (typename REGION_TYPE::Iterator i = region.begin(); i != region.end(); ++i) {
            addCoord(&coordsBuffer, *i, index++, region.size());
        }

        cudaMemcpy(coords.data(), &coordsBuffer[0], size * sizeof(int), cudaMemcpyHostToDevice);
    }

    int *data()
    {
        return coords.data();
    }

    const int *data() const
    {
        return coords.data();
    }

private:

    LibFlatArray::cuda_array<int> coords;

    void addCoord(std::vector<int> *coordsBuffer, const Coord<1>& c, std::size_t index, std::size_t stride) const
    {
        (*coordsBuffer)[0 * stride + index] = c[0];
    }

    void addCoord(std::vector<int> *coordsBuffer, const Coord<2>& c, std::size_t index, std::size_t stride) const
    {
        (*coordsBuffer)[0 * stride + index] = c[0];
        (*coordsBuffer)[1 * stride + index] = c[1];
    }

    void addCoord(std::vector<int> *coordsBuffer, const Coord<3>& c, std::size_t index, std::size_t stride) const
    {
        (*coordsBuffer)[0 * stride + index] = c[0];
        (*coordsBuffer)[1 * stride + index] = c[1];
        (*coordsBuffer)[2 * stride + index] = c[2];
    }
};

}

#endif
