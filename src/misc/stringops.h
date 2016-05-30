#ifndef LIBGEODECOMP_MISC_STRINGOPS_H
#define LIBGEODECOMP_MISC_STRINGOPS_H

#include <libgeodecomp/misc/stringvec.h>
#include <set>
#include <sstream>

// sad but true: CodeGear's C++ compiler has troubles with the +
// operator for strings.
#ifdef __CODEGEARC__
namespace LibGeoDecomp {
std::string operator+(const std::string& a, const std::string& b)
{
    return std::string(a).append(b);
}

}
#endif

namespace LibGeoDecomp {

/**
 * Common string operations which are mysterically missing in C++'s
 * standard library.
 */
class StringOps
{
public:

    static std::string itoa(int i)
    {
        std::stringstream buf;
        buf << i;
        return buf.str();
    }

    static int atoi(const std::string& s)
    {
        std::stringstream buf(s);
        int ret;
        buf >> ret;
        return ret;
    }

    static double atof(const std::string& s)
    {
        std::stringstream buf(s);
        double ret;
        buf >> ret;
        return ret;
    }

    static StringVec tokenize(const std::string& string, const std::string& delimiters)
    {
        StringVec ret;

        std::set<std::size_t> startPositions;
        std::set<std::size_t> endPositions;

        startPositions.insert(0);
        endPositions.insert(string.size());

        for (std::string::const_iterator delimiter = delimiters.begin(); delimiter != delimiters.end(); ++delimiter) {
            std::size_t pos = string.find(*delimiter, 0);

            for (;
                 (pos != std::string::npos) && (pos < string.length());
                 pos = string.find(*delimiter, pos + 1)) {
                endPositions.insert(pos);
                startPositions.insert(pos + 1);
            }
        }

        std::set<std::size_t>::iterator i = startPositions.begin();
        std::set<std::size_t>::iterator j = endPositions.begin();

        for (; i != startPositions.end(); ++i, ++j) {
            if (*i != *j) {
                ret << string.substr(*i, *j - *i);
            }
        }

        return ret;
    }

    static std::string join(const StringVec& tokens, const std::string& delimiter)
    {
        std::stringstream buf;

        for (StringVec::const_iterator i = tokens.begin(); i != tokens.end(); ++i) {
            if (i != tokens.begin()) {
                buf << delimiter;
            }
            buf << *i;
        }

        return buf.str();
    }
};

}

#endif
