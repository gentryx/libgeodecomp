#ifndef LIBGEODECOMP_IO_IOEXCEPTION_H
#define LIBGEODECOMP_IO_IOEXCEPTION_H

// Kill some warnings in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4710 4711 )
#endif

#include <cstring>
#include <stdexcept>

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

// Hardwire this warning to off as MSVC would otherwise complain about
// inline functions not being included in object files:
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 )
#endif

/**
 * exception class which is suitable for all I/O errors. More specific
 * error conditions can be represented as subclasses.
 */
class IOException : public std::runtime_error
{
public:
    /**
     * Initializes a new IOException object which means that an error described
     * by message occured.
     */
    explicit IOException(std::string message) :
        std::runtime_error(message)
    {}

    virtual ~IOException() throw ()
    {}
};


/**
 * More specific error class for errors during file opening.
 */
class FileOpenException : public IOException
{
public:
    explicit FileOpenException(
        std::string file) :
        IOException("Could not open file " + file)
    {}
};


/**
 * More specific error class for errors during writing.
 */
class FileWriteException : public IOException
{
public:
    explicit FileWriteException(
        std::string file) :
        IOException("Could not write file " + file)
    {}
};


/**
 * More specific error class for errors during reading.
 */
class FileReadException : public IOException
{
public:
    explicit FileReadException(
        std::string file) :
        IOException("Could not read file " + file)
    {}
};

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

}

#endif
