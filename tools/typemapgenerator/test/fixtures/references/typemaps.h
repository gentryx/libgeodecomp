#ifndef TYPEMAPS_H
#define TYPEMAPS_H

#include <mpi.h>
#include <complex>
#include <rim.h>
#include <tire.h>
#include <wheel.h>

extern MPI_Datatype MPI_RIM;
extern MPI_Datatype MPI_TIRE;
extern MPI_Datatype MPI_WHEEL;

class Typemaps
{
public:
    static void initializeMaps();

    template<typename T>
    static inline MPI_Datatype lookup()
    {
        return lookup((T*)0);
    }

private:
    template<typename T>
    static MPI_Aint getAddress(T *address)
    {
        MPI_Aint ret;
        MPI_Get_address(address, &ret);
        return ret;
    }

    static MPI_Datatype generateMapRim();
    static MPI_Datatype generateMapTire();
    static MPI_Datatype generateMapWheel();

    static inline MPI_Datatype lookup(bool*)
    {
        return MPI_CHAR;
    }

    static inline MPI_Datatype lookup(char*)
    {
        return MPI_CHAR;
    }

    static inline MPI_Datatype lookup(double*)
    {
        return MPI_DOUBLE;
    }

    static inline MPI_Datatype lookup(float*)
    {
        return MPI_FLOAT;
    }

    static inline MPI_Datatype lookup(int*)
    {
        return MPI_INT;
    }

    static inline MPI_Datatype lookup(long*)
    {
        return MPI_LONG;
    }

    static inline MPI_Datatype lookup(long double*)
    {
        return MPI_LONG_DOUBLE;
    }

    static inline MPI_Datatype lookup(long long*)
    {
        return MPI_LONG_LONG;
    }

    static inline MPI_Datatype lookup(short*)
    {
        return MPI_SHORT;
    }

    static inline MPI_Datatype lookup(signed char*)
    {
        return MPI_SIGNED_CHAR;
    }

    static inline MPI_Datatype lookup(std::complex<double>*)
    {
        return MPI_DOUBLE_COMPLEX;
    }

    static inline MPI_Datatype lookup(std::complex<float>*)
    {
        return MPI_COMPLEX;
    }

    static inline MPI_Datatype lookup(unsigned*)
    {
        return MPI_UNSIGNED;
    }

    static inline MPI_Datatype lookup(unsigned char*)
    {
        return MPI_UNSIGNED_CHAR;
    }

    static inline MPI_Datatype lookup(unsigned long*)
    {
        return MPI_UNSIGNED_LONG;
    }

    static inline MPI_Datatype lookup(unsigned long long*)
    {
        return MPI_UNSIGNED_LONG_LONG;
    }

    static inline MPI_Datatype lookup(unsigned short*)
    {
        return MPI_UNSIGNED_SHORT;
    }

    static inline MPI_Datatype lookup(wchar_t*)
    {
        return MPI_WCHAR;
    }

    static inline MPI_Datatype lookup(Rim*)
    {
        return MPI_RIM;
    }

    static inline MPI_Datatype lookup(Tire*)
    {
        return MPI_TIRE;
    }

    static inline MPI_Datatype lookup(Wheel*)
    {
        return MPI_WHEEL;
    }

};

#endif
