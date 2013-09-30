#ifndef LIBGEODECOMP_MISC_SUPERVECTOR_H
#define LIBGEODECOMP_MISC_SUPERVECTOR_H

#include <algorithm>
#include <numeric>
#include <iterator>
#include <sstream>
#include <vector>
#include <libgeodecomp/misc/outputpairs.h>

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_BOOST_SERIALIZATION
#include <boost/serialization/vector.hpp>
#endif

namespace LibGeoDecomp {

using std::vector;

/**
 * Deletes items from @param vec that are equal to @param obj
 */
template <typename T, typename Allocator, typename U>
inline void del(std::vector<T, Allocator> & vec, const U& obj)
{
    vec.erase(std::remove(vec.begin(), vec.end(), obj), vec.end());
}

template <typename T, typename Allocator>
inline std::string toString(const std::vector<T, Allocator>& vec)
{
    std::ostringstream temp;
    temp << "[";
    for (typename std::vector<T, Allocator>::const_iterator i = vec.begin(); i != vec.end();) {
        temp << *i;
        i++;
        if (i != vec.end()) {
            temp << ", ";
        }
    }
    temp << "]";
    return temp.str();
}

template <typename T, typename Allocator>
inline void append(std::vector<T, Allocator>& target, const std::vector<T, Allocator>& other)
{
    target.insert(target.end(), other.begin(), other.end());
}

template <typename T, typename Allocator>
inline void push_front(std::vector<T, Allocator>& vec, const T& obj)
{
    vec.insert(vec.begin(), obj);
}

template <typename T, typename Allocator>
inline T pop_front(std::vector<T, Allocator>& vec)
{
    T ret = vec.front();
    vec.erase(vec.begin());
    return ret;
}

template <typename T, typename Allocator>
inline T pop(std::vector<T, Allocator>& vec)
{
    T ret = vec.back();
    vec.pop_back();
    return ret;
}


template <typename T, typename Allocator>
inline T sum(const std::vector<T, Allocator>& vec)
{
    return std::accumulate(vec.begin(), vec.end(), T());

}

template <typename T, typename Allocator>
inline bool contains(const std::vector<T, Allocator>& vec, const T& element)
{
    return std::find(vec.begin(), vec.end(), element) != vec.end();
}

template <typename T, typename Allocator>
inline void sort(std::vector<T, Allocator>& vec)
{
    std::sort(vec.begin(), vec.end());
}

template <typename T, typename Allocator>
T& (max)(std::vector<T, Allocator>& vec)
{
    return *(std::max_element(vec.begin(), vec.end()));
}

template <typename T, typename Allocator>
const T& (max)(const std::vector<T, Allocator>& vec)
{
    return *(std::max_element(vec.begin(), vec.end()));
}

template <typename T, typename Allocator, typename U>
inline std::vector<T, Allocator>& operator<<(std::vector<T, Allocator>& vec, const U& obj)
{
    vec.push_back(obj);
    return vec;
}

template <typename T, typename Allocator>
inline std::vector<T, Allocator> operator+(std::vector<T, Allocator>& target, const std::vector<T, Allocator>& other)
{
    std::vector<T, Allocator> ret(target);
    append(ret, other);
    return ret;
}

template<typename _CharT, typename _Traits, typename T, typename Allocator>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& os,
           const std::vector<T, Allocator>& vec)
{
    os << toString(vec);
    return os;
}
}

#endif
