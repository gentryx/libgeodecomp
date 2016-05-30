#ifndef LIBGEODECOMP_IO_TIME_H
#define LIBGEODECOMP_IO_TIME_H

#include <ctime>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <sys/time.h>

namespace LibGeoDecomp {

class Time
{
public:
    static double getTime()
    {
        timeval t;
        gettimeofday(&t, 0);

        return t.tv_sec + t.tv_usec * 1.0e-6;
    }

    static std::string renderISO(double time)
    {
        timeval secondsSinceEpoch;
        double intFraction;
        secondsSinceEpoch.tv_usec = std::modf(time, &intFraction) * 1.0e6;
        secondsSinceEpoch.tv_sec = intFraction;
        tm timeSpec;
        localtime_r(&secondsSinceEpoch.tv_sec, &timeSpec);
        char buf[1024];
        strftime(buf, 1024, "%Y.%m.%d %H:%M:%S", &timeSpec);

        std::stringstream stream;
        stream  << buf << "." << std::setw(6) << std::setfill('0') << secondsSinceEpoch.tv_usec;

        return stream.str();
    }
};

}

#endif
