#ifndef _libgeodecomp_misc_superset_h_
#define _libgeodecomp_misc_superset_h_

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

    inline SuperSet() {};

    inline std::string toString() const {
        std::ostringstream temp;
        temp << "{";
        for (const_iterator i = this->begin(); i != this->end();) {
            temp << *i;
            i++;
            if (i != this->end())
                temp << "\n";
        }
        temp << "}";
        return temp.str();
    }

    inline const T& min() const
    {
        return *this->begin();
    }

    inline const T& max() const
    {
        return *this->rbegin();
    }

    inline void erase_min()
    {
        this->erase(this->begin());
    }

    inline SuperSet& operator<<(const T& elem) 
    {
        this->insert(elem);
        return *this;
    }

    inline SuperSet operator&&(const SuperSet<T>& other) const 
    {        
        SuperSet result;
        std::set_intersection(
            this->begin(), this->end(), 
            other.begin(), other.end(), 
            std::inserter(result, result.begin()));
        return result;
    }

    inline SuperSet operator||(const SuperSet<T>& other) const 
    {        
        SuperSet result;
        std::set_union(
            this->begin(), this->end(), 
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
