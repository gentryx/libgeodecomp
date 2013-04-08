#ifndef LIBGEODECOMP_MISC_OUTPUTPAIRS_H

template<typename _CharT, typename _Traits, typename _T1, typename _T2>
std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os,
           const std::pair<_T1, _T2>& p)
{
    __os << "(" << p.first << ", " << p.second << ")";
    return __os;
}

#endif
