#ifndef LIBGEODECOMP_MISC_SUPERMAP_H
#define LIBGEODECOMP_MISC_SUPERMAP_H

#include <iterator>
#include <map>
#include <sstream>

namespace LibGeoDecomp {

/**
 * This class adds some functionality the std::map ought to
 * provide (but fails to).
 */
template<typename Key, typename Value>
class SuperMap : public std::map<Key, Value>
{
public:
    typedef typename std::map<Key, Value>::iterator iterator;
    typedef typename std::map<Key, Value>::const_iterator const_iterator;

    using std::map<Key, Value>::begin;
    using std::map<Key, Value>::end;

    inline SuperMap() {};

    inline const Value& operator[](const Key& key) const
    {
        return (*(const_cast<SuperMap<Key, Value>*>(this)))[key];
    }

    // C++ weirdness: we have to explicitly reimplement this method as
    // in templates overloaded methods in templates hide inherited
    // ones.
    inline Value& operator[](const Key& key)
    {
        return (*((std::map<Key, Value>*)this))[key];
    }

    inline std::string toString() const 
    {
        std::ostringstream temp;
        temp << "{";

        for (const_iterator i = begin(); i != end();) {
            temp << i->first << " => " << i->second;
            i++;
            if (i != end()) {
                temp << ", ";
            }
        }
        temp << "}";
        return temp.str();
    }
};

}

template<typename _CharT, typename _Traits, typename Key, typename Value>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const LibGeoDecomp::SuperMap<Key, Value>& superMap)
{
    __os << superMap.toString();
    return __os;
}

#endif
