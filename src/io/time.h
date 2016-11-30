#ifndef LIBGEODECOMP_IO_TIME_H
#define LIBGEODECOMP_IO_TIME_H

#include <cmath>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace LibGeoDecomp {

/**
 * This class helps with rendering time stamps. It replaces
 * our previous use of boost::posix_time.
 */
class Time
{
public:
    static std::string renderISO(double time)
    {
        double intFraction;
        int uSecondsSinceEpoch = std::modf(time, &intFraction) * 1.0e6;
        time_t secondsSinceEpoch = intFraction;
        tm timeSpec;
        gmtime_r(&secondsSinceEpoch, &timeSpec);
        char buf[1024];
        strftime(buf, 1024, "%Y.%m.%d %H:%M:%S", &timeSpec);

        std::stringstream stream;
        stream  << buf << "." << std::setw(6) << std::setfill('0') << uSecondsSinceEpoch;

        return stream.str();
    }

    static std::string renderDuration(double duration)
    {
        std::stringstream stream;
        double realSeconds;
        double subseconds = std::modf(duration, &realSeconds);

        int totalSeconds = realSeconds;
        int seconds = totalSeconds % 60;
        int minutes = totalSeconds / 60 % 60;
        int hours = totalSeconds / 3600;

        std::stringstream buf;
        buf << std::setw(2) << std::setfill('0') << hours << ":"
            << std::setw(2) << std::setfill('0') << minutes << ":"
            << std::setw(2) << std::setfill('0') << seconds;

        if (subseconds > 0) {
            int fraction = subseconds * 1.0e6;
            buf << "." << std::setw(6) << fraction;
        }

        return buf.str();
    }
};

}

#endif
