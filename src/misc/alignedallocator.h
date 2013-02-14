#ifndef _libgeodecomp_misc_alignedallocator_h_
#define _libgeodecomp_misc_alignedallocator_h_

#ifdef __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif

namespace LibGeoDecomp {

template<class T, size_t ALIGNMENT>
class AlignedAllocator
{
public:
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;

    inline pointer address(reference x) const
    {
        return &x;
    }
    
    inline const_pointer address(const_reference x) const
    {
        return &x;
    }

    pointer allocate(size_type n, const void* =0)
    {
        // This code whould have been a piece of cake if it would have
        // worket with posix_memalign, which it doesn't. Alternatively
        // we allocate a larger chunk of memory in which we can
        // accomodate an array of the selected size, shifted to the
        // desired offset. Since we need the original address for the
        // deallocation, we store it directly in front of the aligned
        // array's start. Ugly, but it works.
        char *chunk = std::allocator<char>().allocate(upsize(n));
        if (chunk == 0) {
            return (pointer)chunk;
        }

        size_type offset = (size_type)chunk % ALIGNMENT;
        size_type correction = ALIGNMENT - offset;
        if (correction < sizeof(char*))
            correction += ALIGNMENT;
        char *ret = chunk + correction;
        *((char**)ret - 1) = chunk;
        return (pointer)ret;
    }

    void deallocate(pointer p, size_type n)
    {
        if (p == 0) {
            return;
        }

        char *actual;
        // retrieve the original pointer which sits in front of its
        // aligned brother
        actual = *((char**)p - 1);
        std::allocator<char>().deallocate(actual, upsize(n));
    }

    size_type max_size() const throw()
    {
        return std::allocator<T>().max_size();
    }

    void construct(pointer p, const_reference val)
    {
        std::allocator<T>().construct(p, val);
    }
    
    void destroy(pointer p)
    {
        std::allocator<T>().destroy(p);
    }

private:
    size_type graceOffset()
    {
        return ALIGNMENT + sizeof(char*);
    }

    size_type upsize(size_type n)
    {
        return n * sizeof(T) + graceOffset();
    }
};

}

#endif
