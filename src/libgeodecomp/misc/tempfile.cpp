#include <libgeodecomp/config.h>

#ifdef LIBGEODECOMP_WITH_MPI
#include <libgeodecomp/communication/mpilayer.h>
#endif

// Kill warning 4514 in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

#include <libgeodecomp/misc/random.h>
#include <libgeodecomp/misc/stringops.h>
#include <libgeodecomp/misc/tempfile.h>

#include <fstream>

#ifdef _WIN32

#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4820 )
#endif

#include <io.h>
#include <stdlib.h>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

#else
#include <unistd.h>
#endif

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

// ugly, but good enough for now
std::string TempFile::serial(const std::string& prefix)
{
    for (;;) {
        std::stringstream buf;
#ifdef _WIN32
        char *tempVar;
        std::size_t length;
        errno_t err = _dupenv_s(&tempVar, &length, "TMP");
        if (err) {
            throw std::runtime_error("could not retrieve value of environment variable TMP");
        }
        buf << tempVar
            << '\\';
#else
        const char* tempDir = getenv("TMPDIR");
        buf << (tempDir? tempDir : "/tmp")
            << '/';
#endif
        unsigned r = Random::genUnsigned();
        buf << prefix
            << StringOps::itoa(r);

        std::string name = buf.str();

        if (!(exists(name))) {
            return name;
        }
    }
}

#ifdef LIBGEODECOMP_WITH_MPI

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

bool TempFile::exists(const std::string& filename)
{
#ifdef _WIN32
    int accessResult = _access(filename.c_str(), 0);
#else
    int accessResult = access(filename.c_str(), F_OK);
#endif

    return (accessResult == 0);
}

void TempFile::unlink(const std::string& filename)
{
#ifdef _WIN32
    _unlink(filename.c_str());
#else
    ::unlink(filename.c_str());
#endif
}

}

#ifdef _MSC_BUILD
#pragma warning( disable : 4710 )
#endif
