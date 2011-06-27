#include <string.h>
#include <libgeodecomp/io/ioexception.h>
#include <libgeodecomp/misc/stringops.h>

namespace LibGeoDecomp {

IOException::IOException(
    std::string msg, 
    std::string file, 
    int error,
    bool fatal)
  : std::runtime_error(msg), _file(file), _error(error), _fatal(fatal) 
{}

std::string IOException::file()
{
    return _file;
}

int IOException::error()
{
    return _error;
}

bool IOException::fatal()
{
    return _fatal;
}

std::string IOException::toString()
{
    std::string out(std::string(what()) + " `" + _file + "'");
    if (_error != 0) out += std::string(": ") + strerror(_error);
    return out;
}

};


