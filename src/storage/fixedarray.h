#ifndef LIBGEODECOMP_STORAGE_FIXEDARRAY_H
#define LIBGEODECOMP_STORAGE_FIXEDARRAY_H

#include <stdexcept>
#include <libgeodecomp/config.h>

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
    friend class Serialization;
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

    FixedArray<T, SIZE>& operator-=(const FixedArray<T, SIZE>& other)
    {
        std::size_t minSize = std::min(elements, other.elements);
        std::size_t maxSize = std::max(elements, other.elements);
        std::size_t i = 0;

        for (; i < minSize; ++i) {
            store[i] -= other[i];
        }

        for (; i < other.size(); ++i) {
            store[i] = -other[i];
        }

        elements = maxSize;
        return *this;
    }

    FixedArray<T, SIZE>& operator+=(const FixedArray<T, SIZE>& other)
    {
        std::size_t minSize = std::min(elements, other.elements);
        std::size_t maxSize = std::max(elements, other.elements);
        std::size_t i = 0;

        for (; i < minSize; ++i) {
            store[i] += other[i];
        }

        for (; i < other.size(); ++i) {
            store[i] = other[i];
        }

        elements = maxSize;
        return *this;
    }

    FixedArray<T, SIZE> operator-(const FixedArray<T, SIZE>& other) const
    {
        std::size_t minSize = std::min(elements, other.elements);
        std::size_t maxSize = std::max(elements, other.elements);
        FixedArray<T, SIZE> ret(maxSize);
        std::size_t i = 0;

        for (; i < minSize; ++i) {
            ret[i] = store[i] - other[i];
        }

        for (; i < size(); ++i) {
            ret[i] = store[i];
        }

        for (; i < other.size(); ++i) {
            ret[i] = -other[i];
        }

        return ret;
    }

    FixedArray<T, SIZE> operator+(const FixedArray<T, SIZE>& other) const
    {
        std::size_t minSize = std::min(elements, other.elements);
        std::size_t maxSize = std::max(elements, other.elements);
        FixedArray<T, SIZE> ret(maxSize);
        std::size_t i = 0;

        for (; i < minSize; ++i) {
            ret[i] = store[i] + other[i];
        }

        for (; i < size(); ++i) {
            ret[i] = store[i];
        }

        for (; i < other.size(); ++i) {
            ret[i] = other[i];
        }

        return ret;
    }

    template<typename F>
    FixedArray<T, SIZE>& operator/=(F f)
    {
        for (std::size_t i = 0; i < size(); ++i) {
            (*this)[i] /= f;
        }

        return *this;
    }

    template<typename F>
    FixedArray<T, SIZE>& operator*=(F f)
    {
        FixedArray<T, SIZE> ret(size());

        for (std::size_t i = 0; i < size(); ++i) {
            (*this)[i] *= f;
        }

        return *this;
    }

    template<typename F>
    FixedArray<T, SIZE> operator/(F f) const
    {
        FixedArray<T, SIZE> ret(size());

        for (std::size_t i = 0; i < size(); ++i) {
            ret[i] = (*this)[i] / f;
        }

        return ret;
    }

    template<typename F>
    FixedArray<T, SIZE> operator*(F f) const
    {
        FixedArray<T, SIZE> ret(size());

        for (std::size_t i = 0; i < size(); ++i) {
            ret[i] = (*this)[i] * f;
        }

        return ret;
    }

    bool operator<(const FixedArray<T, SIZE>& other) const
    {
        if (size() != other.size()) {
            return size() < other.size();
        }

        for (std::size_t i = 0; i < size(); ++i) {
            if ((*this)[i] != other[i]) {
                return (*this)[i] < other[i];
            }
        }

        return false;
    }

    bool operator>(const FixedArray<T, SIZE>& other) const
    {
        if (size() != other.size()) {
            return size() > other.size();
        }

        for (std::size_t i = 0; i < size(); ++i) {
            if ((*this)[i] != other[i]) {
                return (*this)[i] > other[i];
            }
        }

        return false;
    }

    bool operator==(const FixedArray<T, SIZE>& other) const
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

    inline std::size_t size() const
    {
        return elements;
    }

private:
    T store[SIZE];
    std::size_t elements;
};

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
