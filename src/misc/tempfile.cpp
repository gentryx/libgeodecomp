#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_FEATURE_MPI
#include <libgeodecomp/mpilayer/mpilayer.h>
#endif

#include <fstream>
#include <boost/filesystem.hpp>
#include <libgeodecomp/misc/random.h>
#include <libgeodecomp/misc/stringops.h>
#include <libgeodecomp/misc/tempfile.h>

namespace LibGeoDecomp {

// ugly, but good enough for now
std::string TempFile::serial(const std::string& prefix)
{
    std::string ret;
    
    for (;;) {            
#ifdef __WIN32__
        std::string name = getenv("TMP");
#else
        std::string name = "/tmp";
#endif
        boost::filesystem::path path(name);
        unsigned r = Random::gen_u();
        path /= prefix + StringConv::itoa(r);
        if (!boost::filesystem::exists(path)) {
            ret = path.string();
            return ret;
        
        }
    }
}

#ifdef LIBGEODECOMP_FEATURE_MPI

std::string TempFile::parallel(const std::string& prefix)
{
    std::string ret;

    if (MPILayer().rank() == 0) {
        ret = serial(prefix);
        int len = ret.size();
        MPI::COMM_WORLD.Bcast(&len, 1, MPI::INT, 0);
        MPI::COMM_WORLD.Bcast((void*)ret.c_str(), len, MPI::CHAR, 0);
    } else {
        int len;
        MPI::COMM_WORLD.Bcast(&len, 1, MPI::INT, 0);
        ret = std::string(len, 'X');
        MPI::COMM_WORLD.Bcast((void*)ret.c_str(), len, MPI::CHAR, 0);
    }

    return ret;
}
#endif

};
