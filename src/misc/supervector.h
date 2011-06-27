#ifndef _libgeodecomp_misc_supervector_h_
#define _libgeodecomp_misc_supervector_h_

#include <algorithm>
#include <iterator>
#include <sstream>
#include <vector>

namespace LibGeoDecomp {

/**
 * This class adds some functionality the std::vector ought to
 * provide (but fails to).
 */
template<typename T, typename Allocator = std::allocator<T> >
class SuperVector : public std::vector<T, Allocator>
{
public:
    typedef typename std::vector<T>::iterator iterator;
    typedef typename std::vector<T>::const_iterator const_iterator;

    inline SuperVector() {}
    inline SuperVector(int i) : std::vector<T>(i) {}
    inline SuperVector(int i, T t) : std::vector<T>(i, t) {}

    /** 
     * Deletes items from _self_ that are equal to @param obj
     */
    inline void del(const T &obj) 
    {
        erase(std::remove(this->begin(), this->end(), obj), this->end());
    }

    // We have to use the inherited operator by hand, as this requires a cast
    inline bool operator==(const SuperVector<T> &comp) const 
    {
        return ((std::vector<T>)*this) == ((std::vector<T>)comp);
    }

    inline std::string toString() const 
    {
        std::ostringstream temp;
        temp << "[";
        for (const_iterator i = this->begin(); i != this->end();) {
            temp << *i;
            i++;
            if (i != this->end())
                temp << ", ";
        }
        temp << "]";
        return temp.str();
    }

    inline SuperVector& operator<<(const T& obj)
    {
        this->push_back(obj);
        return *this;
    }

    inline void push_front(const T& obj) 
    {
        insert(this->begin(), obj);
    }

    inline T pop()
    {
        T ret = this->back();
        this->pop_back();
        return ret;
    }

    inline T sum() const 
    {
        T res = 0;
        for (const_iterator i = this->begin(); i != this->end(); i++) 
            res += *i;
        return res;

    }

    inline SuperVector<T> concat(const SuperVector<T>& other) const 
    {        
        SuperVector<T> ret = *this;
        ret.append(other);
        return ret;
    }

    inline void append(const SuperVector<T>& other) 
    {        
        insert(this->end(), other.begin(), other.end());
    }

    inline bool contains(const T& element) const
    {
        return std::find(this->begin(), this->end(), element) != this->end();
    }

    /**
     * true iff self and other have at least one common member
     */
    inline bool hasCommonElement(const SuperVector<T>& other) const
    {
        for (const_iterator i = this->begin(); i != this->end(); i++)
            if (other.contains(*i)) return true;
        return false;
    }

    inline void sort()
    {
        std::sort(this->begin(), this->end());
    }

    T& max()
    {
        return *(std::max_element(this->begin(), this->end()));
    }

    const T& max() const
    {
        return *(std::max_element(this->begin(), this->end()));
    }
};

}

template<typename _CharT, typename _Traits, typename T>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const LibGeoDecomp::SuperVector<T>& superVector)
{
    __os << superVector.toString();
    return __os;
}

#endif
