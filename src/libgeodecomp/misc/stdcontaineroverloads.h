#ifndef LIBGEODECOMP_MISC_STDCONTAINEROVERLOADS_H
#define LIBGEODECOMP_MISC_STDCONTAINEROVERLOADS_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX
#include <hpx/config.hpp>
#endif
#ifdef LIBGEODECOMP_WITH_BOOST_SERIALIZATION
#include <boost/serialization/vector.hpp>
#endif
#ifdef LIBGEODECOMP_WITH_CPP14
#include <utility>
#endif

#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable :  4548 )
#endif

#include <algorithm>
#include <deque>
#include <iterator>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <vector>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

/**
 * vector:
 */

/**
 * Deletes items from @param vec that are equal to @param obj
 */
template <typename T, typename Allocator, typename U>
inline void del(std::vector<T, Allocator>&  vec, const U& obj)
{
    vec.erase(std::remove(vec.begin(), vec.end(), obj), vec.end());
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
T& (min)(std::vector<T, Allocator>& vec)
{
    return *(std::min_element(vec.begin(), vec.end()));
}

template <typename T, typename Allocator>
const T& (min)(const std::vector<T, Allocator>& vec)
{
    return *(std::min_element(vec.begin(), vec.end()));
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

#ifdef LIBGEODECOMP_WITH_CPP14

template <typename T, typename Allocator, typename U>
inline std::vector<T, Allocator>& operator<<(std::vector<T, Allocator>& vec, U&& obj)
{
    vec.push_back(std::forward<U>(obj));
    return vec;
}

#endif

template <typename T, typename Allocator>
inline std::vector<T, Allocator> operator+(const std::vector<T, Allocator>& source1, const std::vector<T, Allocator>& source2)
{
    std::vector<T, Allocator> ret(source1);
    append(ret, source2);
    return ret;
}

template <template<int> class COORD, int DIM>
inline std::vector<typename COORD<DIM>::ValueType> toVector(const COORD<DIM>& coord)
{
    std::vector<typename COORD<DIM>::ValueType> vec;

    for (int i = 0; i < DIM; ++i) {
        vec << coord[i];
    }

    return vec;
}

/**
 * set
 */
template <typename T, typename Allocator, typename U>
inline std::set<T, Allocator>& operator<<(std::set<T, Allocator>& set, const U& obj)
{
    set.insert(obj);
    return set;
}

template <typename T, typename Allocator>
const T& (min)(const std::set<T, Allocator>& set)
{
    return *set.begin();
}

template <typename T, typename Allocator>
const T& (max)(const std::set<T, Allocator>& set)
{
    return *set.rbegin();
}

template <typename T, typename Allocator>
void erase_min(std::set<T, Allocator>& set)
{
    set.erase(set.begin());
}

template <typename T, typename Allocator>
inline std::set<T, Allocator> operator&&(
    const std::set<T, Allocator>& set,
    const std::set<T, Allocator>& other)
{
    std::set<T, Allocator> result;
    std::set_intersection(
        set.begin(), set.end(),
        other.begin(), other.end(),
        std::inserter(result, result.begin()));
    return result;
}

template <typename T, typename Allocator>
inline std::set<T, Allocator> operator||(
    const std::set<T, Allocator>& set,
    const std::set<T, Allocator>& other)
{
    std::set<T, Allocator> result;
    std::set_union(
        set.begin(), set.end(),
        other.begin(), other.end(),
        std::inserter(result, result.begin()));
    return result;
}

template <typename T, typename Allocator>
inline std::set<T, Allocator>& operator|=(
    std::set<T, Allocator>& set,
    const std::set<T, Allocator>& other)
{
    set.insert(other.begin(), other.end());
    return set;
}

template <typename T, typename Allocator>
inline std::set<T, Allocator> operator+(
    const std::set<T, Allocator>& set,
    const std::set<T, Allocator>& other)
{
    std::set<T, Allocator> ret = set;
    ret |= other;
    return ret;
}

template <typename T, typename Allocator>
inline std::set<T, Allocator>& operator-=(
    std::set<T, Allocator>& set,
    const std::set<T, Allocator>& other)
{
    for (typename std::set<T, Allocator>::const_iterator i = other.begin();
         i != other.end();
         ++i) {
        set.erase(*i);
    }
    return set;
}

template <typename T, typename Allocator>
inline std::set<T, Allocator> operator-(
    const std::set<T, Allocator>& set,
    const std::set<T, Allocator>& other)
{
    std::set<T, Allocator> ret = set;
    ret -= other;
    return ret;
}

/**
 * deque
 */
template <typename T, typename Allocator>
inline void append(std::deque<T, Allocator>& target, const std::deque<T, Allocator>& other)
{
    target.insert(target.end(), other.begin(), other.end());
}

template <typename T, typename Allocator, typename U>
inline std::deque<T, Allocator>& operator<<(std::deque<T, Allocator>& deque, const U& obj)
{
    deque.push_back(obj);
    return deque;
}

template <typename T, typename Allocator>
inline std::deque<T, Allocator> operator+(
    const std::deque<T, Allocator>& source1, const std::deque<T, Allocator>& source2)
{
    std::deque<T, Allocator> ret(source1);
    append(ret, source2);
    return ret;
}

template <typename T, typename Allocator>
inline T sum(const std::deque<T, Allocator>& vec)
{
    return std::accumulate(vec.begin(), vec.end(), T());

}

template <typename T, typename Allocator>
inline bool contains(const std::deque<T, Allocator>& vec, const T& element)
{
    return std::find(vec.begin(), vec.end(), element) != vec.end();
}

template <typename T, typename Allocator>
inline void sort(std::deque<T, Allocator>& vec)
{
    std::sort(vec.begin(), vec.end());
}

template <typename T, typename Allocator>
T& (min)(std::deque<T, Allocator>& vec)
{
    return *(std::min_element(vec.begin(), vec.end()));
}

template <typename T, typename Allocator>
const T& (min)(const std::deque<T, Allocator>& vec)
{
    return *(std::min_element(vec.begin(), vec.end()));
}

template <typename T, typename Allocator>
T& (max)(std::deque<T, Allocator>& vec)
{
    return *(std::max_element(vec.begin(), vec.end()));
}

template <typename T, typename Allocator>
const T& (max)(const std::deque<T, Allocator>& vec)
{
    return *(std::max_element(vec.begin(), vec.end()));
}

/**
 * Output
 */
template<typename _CharT, typename _Traits, typename _T1, typename _T2>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& os,
           const std::pair<_T1, _T2>& p)
{
    os << "(" << p.first << ", " << p.second << ")";
    return os;
}

template<typename _CharT, typename _Traits, typename T, typename Allocator>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& os,
           const std::vector<T, Allocator>& vec)
{
    os << "[";

    if (vec.size()) {
        typename std::vector<T, Allocator>::const_iterator i = vec.begin();
        os << *i;
        ++i;

        for (; i != vec.end(); ++i) {
            os << ", " << *i;
        }
    }

    os << "]";

    return os;
}

template<typename _CharT, typename _Traits, typename Key, typename Value, typename Allocator>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& os,
           const std::map<Key, Value, Allocator>& map)
{
    os << "{";

    if (map.size()) {
        typename std::map<Key, Value, Allocator>::const_iterator i = map.begin();
        os << i->first << " => " << i->second;
        ++i;

        for (; i != map.end(); ++i) {
            os << ", " << i->first << " => " << i->second;
        }

    }

    os << "}";

    return os;
}

template<typename _CharT, typename _Traits, typename T, typename Allocator>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& os,
           const std::set<T, Allocator>& set)
{
    os << "{";

    if (set.size()) {
        typename std::set<T, Allocator>::const_iterator i = set.begin();
        os << *i;
        ++i;

        for (; i != set.end(); ++i) {
            os << ", " << *i;
        }
    }

    os << "}";

    return os;
}

template<typename _CharT, typename _Traits, typename T, typename Allocator>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& os,
           const std::deque<T, Allocator>& deque)
{
    os << "(";

    if (deque.size()) {
        typename std::deque<T, Allocator>::const_iterator i = deque.begin();
        os << *i;
        ++i;

        for (; i != deque.end(); ++i) {
            os << ", " << *i;
        }
    }

    os << ")";

    return os;
}

#ifdef LIBGEODECOMP_WITH_CUDA
#ifdef __CUDACC__

template<typename _CharT, typename _Traits>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& os,
           dim3 dim)
{
    os << "(" << dim.x << ", " << dim.y << ", " << dim.z << ")";
    return os;
}

#endif
#endif

}




#endif
