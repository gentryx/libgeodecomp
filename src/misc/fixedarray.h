#ifndef _libgeodecomp_misc_fixedarray_h_
#define _libgeodecomp_misc_fixedarray_h_

#include <stdexcept>

namespace boost {
namespace serialization {

class access;

}
}

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
    friend class boost::serialization::access;

    FixedArray(const int& _elements=0) :
        elements(_elements)
    {}

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

    void erase(const T *elem)
    {
        for (T *i = const_cast<T*>(elem); i != end(); ++i) {
            *i = *(i + 1);
        }

        --elements;
    }

    void reserve(const int& num)
    {
        if (num > SIZE) {
            throw std::out_of_range("reserving too many elements");
        }

        elements = num;
    }

    template<class Archive>
    inline void serialize(Archive &archive, const unsigned int version)
    {
        archive & store;
        archive & elements;
    }

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
