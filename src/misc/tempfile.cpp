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
    for (;;) {
        std::stringstream buf;
#ifdef __WIN32__
        std::string name = getenv("TMP");
        buf << '\\';
#else
        const char* tempDir = getenv("TMPDIR");
        std::string name = tempDir? tempDir : "/tmp";
        buf << '/';
#endif
        unsigned r = Random::gen_u();
        std::string filename = prefix + StringOps::itoa(r);
        buf << filename;
        name += buf.str();
        if (!boost::filesystem::exists(name)) {
            return name;
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
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast((void*)ret.c_str(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
    } else {
        int len;
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        ret = std::string(len, 'X');
        MPI_Bcast((void*)ret.c_str(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    return ret;
}

#endif

}
