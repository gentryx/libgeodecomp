#ifndef LIBGEODECOMP_IO_LOGGER_H
#define LIBGEODECOMP_IO_LOGGER_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_HPX
#include <hpx/config.hpp>
#endif

#include <iomanip>
#include <libgeodecomp/io/timestringconversion.h>
#include <libgeodecomp/misc/scopedtimer.h>

namespace LibGeoDecomp {

/**
 * Logger provides a set of functions and macros to selectively print
 * different amounts/levels of debugging information. Filtering
 * happens at compile time, thus no runtime overhead is incurred for
 * filtered messages.
 */

class Logger {
public:
    static const int FATAL = 0;
    static const int FAULT = 1;
    static const int WARN  = 2;
    static const int INFO  = 3;
    static const int DBG   = 4;
};

}

#if LIBGEODECOMP_DEBUG_LEVEL < 0
#define LOG(LEVEL, MESSAGE)
#endif

#if LIBGEODECOMP_DEBUG_LEVEL >= 0
#define LOG(LEVEL, MESSAGE)                                             \
    if ((LibGeoDecomp::Logger::LEVEL) <= LIBGEODECOMP_DEBUG_LEVEL) {    \
        std::cout << #LEVEL[0] << ", ["                                 \
                  << TimeStringConversion::renderISO(ScopedTimer::time()) \
                  << "] " << std::right                                 \
                  << std::setw(5) << #LEVEL                             \
                  << " -- " << MESSAGE << "\n";                         \
    }
#endif

#endif
