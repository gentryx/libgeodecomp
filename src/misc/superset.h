#ifndef LIBGEODECOMP_MISC_SUPERSET_H
#define LIBGEODECOMP_MISC_SUPERSET_H

#include <algorithm>
#include <iterator>
#include <set>
#include <sstream>

namespace LibGeoDecomp {

/**
 * This class adds some functionality the std::set ought to
 * provide (but fails to).
 */
template<typename T>
class SuperSet : public std::set<T>
{
public:
    typedef typename std::set<T>::iterator iterator;
    typedef typename std::set<T>::const_iterator const_iterator;

    using std::set<T>::begin;
    using std::set<T>::end;
    using std::set<T>::erase;
    using std::set<T>::insert;
    using std::set<T>::rbegin;

    inline SuperSet() {};

    inline std::string toString() const {
        std::ostringstream temp;
        temp << "{";
        for (const_iterator i = begin(); i != end();) {
            temp << *i;
            i++;
            if (i != end())
                temp << "\n";
        }
        temp << "}";
        return temp.str();
    }

    inline const T& min() const
    {
        return *begin();
    }

    inline const T& max() const
    {
        return *rbegin();
    }

    inline void erase_min()
    {
        erase(begin());
    }

    inline SuperSet& operator<<(const T& elem) 
    {
        insert(elem);
        return *this;
    }

    inline SuperSet operator&&(const SuperSet<T>& other) const 
    {        
        SuperSet result;
        std::set_intersection(
            begin(), end(), 
            other.begin(), other.end(), 
            std::inserter(result, result.begin()));
        return result;
    }

    inline SuperSet operator||(const SuperSet<T>& other) const 
    {        
        SuperSet result;
        std::set_union(
            begin(), end(), 
            other.begin(), other.end(), 
            std::inserter(result, result.begin()));
        return result;
    }
};

};

template<typename _CharT, typename _Traits, typename T>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const LibGeoDecomp::SuperSet<T>& superSet)
{
    __os << superSet.toString();
    return __os;
}

#endif
