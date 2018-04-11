#ifndef LIBGEODECOMP_STORAGE_FIXEDARRAY_H
#define LIBGEODECOMP_STORAGE_FIXEDARRAY_H

#include <libgeodecomp/config.h>

// Kill some warnings in system headers:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4996 )
#endif

#include <algorithm>
#include <stdexcept>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4820 )
#endif

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
    friend class BoostSerialization;
    friend class HPXSerialization;
    friend class Typemaps;
    typedef T* iterator;
    typedef const T* const_iterator;
    typedef T value_type;

    explicit FixedArray(std::size_t elements = 0) :
        elements(elements)
    {}

    FixedArray(std::size_t elements, const T& value) :
        elements(elements)
    {
        std::fill(begin(), end(), value);
    }

    inline T& operator[](const std::size_t& i)
    {
        return store[i];
    }

    inline const T& operator[](const std::size_t& i) const
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

    inline static std::size_t capacity()
    {
        return SIZE;
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

    FixedArray<T, SIZE> operator+(const FixedArray<T, SIZE>& other) const
    {
        if ((elements + other.elements) > SIZE) {
            throw std::out_of_range("FixedArray capacity exceeded in concatenation");
        }

        FixedArray<T, SIZE> ret;

        for (std::size_t i = 0; i < elements; ++i) {
            ret << (*this)[i];
        }

        for (std::size_t i = 0; i < other.elements; ++i) {
            ret << other[i];
        }

        return ret;
    }

    FixedArray<T, SIZE>& operator+=(const FixedArray<T, SIZE>& other)
    {
        if ((elements + other.elements) > SIZE) {
            throw std::out_of_range("FixedArray capacity exceeded in concatenation");
        }

        for (std::size_t i = 0; i < other.elements; ++i) {
            (*this) << other[i];
        }

        return *this;
    }

    template<int SIZE2>
    bool operator==(const FixedArray<T, SIZE2>& other) const
    {
        if (size() != other.size()) {
            return false;
        }

        for (std::size_t i = 0; i < size(); ++i) {
            if ((*this)[i] != other[i]) {
                return false;
            }
        }

        return true;
    }

    bool operator!=(const FixedArray<T, SIZE>& other) const
    {
        return !(*this == other);
    }

    void clear()
    {
        elements = 0;
    }

    void erase(T *elem)
    {
        for (T *i = elem; i != (end() - 1); ++i) {
            *i = *(i + 1);
        }

        --elements;
    }

    void remove(std::size_t index)
    {
        for (T *i = begin() + index; i != (end() - 1); ++i) {
            *i = *(i + 1);
        }

        --elements;
    }

    void reserve(std::size_t num)
    {
        if (num > SIZE) {
            throw std::out_of_range("reserving too many elements");
        }

        // this is a NOP since we don't reallocate
    }

    void resize(std::size_t num)
    {
        reserve(num);
        elements = num;
    }

    inline std::size_t size() const
    {
        return elements;
    }

private:
    T store[SIZE];
    std::size_t elements;
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

template<typename CharT, typename Traits, typename T, int N>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os,
           const FixedArray<T, N>& a)
{
    os << "(";

    if (a.size() > 0) {
        os << a[0];

        for (std::size_t i = 1; i < a.size(); ++i) {
            os << ", " << a[i];
        }
    }

    os << ")";
    return os;
}

}

#endif
