#ifndef LIBGEODECOMP_MISC_FIXEDARRAY_H
#define LIBGEODECOMP_MISC_FIXEDARRAY_H

#include <stdexcept>
#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
#include <boost/serialization/base_object.hpp>
#endif

namespace LibGeoDecomp {

/**
 * This is an alternative array with a fixed maximum size. Good for
 * use within objects which should not contain pointers (and are thus
 * serializable by simply copying them bitwise), e.g. simulation
 * cells.
 */
template<typename T, int SIZE>
class FixedArray
{
public:
    friend class Typemaps;

#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
    friend class boost::serialization::access;
#endif

    FixedArray(int elements = 0) :
        elements(elements)
    {}

    FixedArray(int elements, const T& value) :
        elements(elements)
    {
        std::fill(begin(), end(), value);
    }

    typedef T* iterator;
    inline T& operator[](const int& i)
    {
        return store[i];
    }

    inline const T& operator[](const int& i) const
    {
        return store[i];
    }

    T *begin()
    {
        return store;
    }

    const T *begin() const
    {
        return store;
    }

    T *end()
    {
        return store + elements;
    }

    const T *end() const
    {
        return store + elements;
    }

    void push_back(const T& t)
    {
        if (elements >= SIZE) {
            throw std::out_of_range("capacity exceeded");
        }

        store[elements++] = t;
    }

    FixedArray<T, SIZE>& operator<<(const T& t)
    {
        push_back(t);
        return *this;
    }

    void clear()
    {
        elements = 0;
    }

    void erase(T *elem)
    {
        for (T *i = elem; i != end(); ++i) {
            *i = *(i + 1);
        }

        --elements;
    }

    void reserve(int num)
    {
        if (num > SIZE) {
            throw std::out_of_range("reserving too many elements");
        }

        elements = num;
    }

#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
    template<class Archive>
    inline void serialize(Archive& archive, const unsigned int version)
    {
        archive & store;
        archive & elements;
    }
#endif

    inline const int& size() const
    {
        return elements;
    }

private:
    T store[SIZE];
    int elements;
};

}

#endif
