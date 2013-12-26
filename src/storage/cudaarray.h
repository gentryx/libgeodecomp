#ifndef LIBGEODECOMP_STORAGE_CUDAARRAY_H
#define LIBGEODECOMP_STORAGE_CUDAARRAY_H

#include <libflatarray/cuda_allocator.hpp>
#include <cuda.h>

namespace LibGeoDecomp {

/**
 * Handles memory allocation and data transfer (intra and inter GPU)
 * on CUDA-capable NVIDIA GPUs.
 */
template<typename ELEMENT_TYPE>
class CUDAArray
{
public:
    inline CUDAArray(std::size_t size = 0) :
        size(size),
        dataPointer(LibFlatArray::cuda_allocator<ELEMENT_TYPE>().allocate(size))
    {}

    inline CUDAArray(const CUDAArray& array) :
        size(array.size),
        dataPointer(LibFlatArray::cuda_allocator<ELEMENT_TYPE>().allocate(array.size))
    {
        cudaMemcpy(dataPointer, array.dataPointer, byteSize(), cudaMemcpyDeviceToDevice);
    }

    inline CUDAArray(const ELEMENT_TYPE *hostData, std::size_t size) :
        size(size),
        dataPointer(LibFlatArray::cuda_allocator<ELEMENT_TYPE>().allocate(size))
    {
        cudaMemcpy(dataPointer, hostData, byteSize(), cudaMemcpyHostToDevice);
    }

    inline CUDAArray(const std::vector<ELEMENT_TYPE>& hostVector) :
        size(hostVector.size()),
        dataPointer(LibFlatArray::cuda_allocator<ELEMENT_TYPE>().allocate(hostVector.size()))
    {
        cudaMemcpy(dataPointer, &hostVector.front(), byteSize(), cudaMemcpyHostToDevice);
    }

    inline ~CUDAArray()
    {
        LibFlatArray::cuda_allocator<ELEMENT_TYPE>().deallocate(dataPointer, size);
    }

    inline void operator=(const CUDAArray& array)
    {
        LibFlatArray::cuda_allocator<ELEMENT_TYPE>().deallocate(dataPointer, size);

        size = array.size;
        dataPointer = LibFlatArray::cuda_allocator<ELEMENT_TYPE>().allocate(size);
        cudaMemcpy(dataPointer, array.dataPointer, byteSize(), cudaMemcpyDeviceToDevice);
    }

    inline void load(const ELEMENT_TYPE *hostData)
    {
        cudaMemcpy(dataPointer, hostData, byteSize(), cudaMemcpyHostToDevice);
    }

    inline void save(ELEMENT_TYPE *hostData) const
    {
        cudaMemcpy(hostData, dataPointer, byteSize(), cudaMemcpyDeviceToHost);
    }

    inline std::size_t byteSize() const
    {
        return size * sizeof(ELEMENT_TYPE);
    }

    __host__ __device__
    inline ELEMENT_TYPE *data()
    {
        return dataPointer;
    }

    __host__ __device__
    inline const ELEMENT_TYPE *data() const
    {
        return dataPointer;
    }

private:
    std::size_t size;
    ELEMENT_TYPE *dataPointer;
};

}

#endif
