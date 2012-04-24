#ifndef _typemaps_h_
#define _typemaps_h_

#include <complex>
#include <mpi.h>
#include <rim.h>
#include <tire.h>
#include <wheel.h>

namespace MPI {
    extern Datatype RIM;
    extern Datatype TIRE;
    extern Datatype WHEEL;
}

class Typemaps {
public:
    static void initializeMaps();

    template<typename T>
    static inline MPI::Datatype lookup() {
        return lookup((T*)0);
    }

private:
    static MPI::Datatype generateMapRim();
    static MPI::Datatype generateMapTire();
    static MPI::Datatype generateMapWheel();

    static inline MPI::Datatype lookup(bool*) { return MPI::BOOL; }
    static inline MPI::Datatype lookup(char*) { return MPI::CHAR; }
    static inline MPI::Datatype lookup(double*) { return MPI::DOUBLE; }
    static inline MPI::Datatype lookup(float*) { return MPI::FLOAT; }
    static inline MPI::Datatype lookup(int*) { return MPI::INT; }
    static inline MPI::Datatype lookup(long*) { return MPI::LONG; }
    static inline MPI::Datatype lookup(long double*) { return MPI::LONG_DOUBLE; }
    static inline MPI::Datatype lookup(long long*) { return MPI::LONG_LONG; }
    static inline MPI::Datatype lookup(short*) { return MPI::SHORT; }
    static inline MPI::Datatype lookup(signed char*) { return MPI::SIGNED_CHAR; }
    static inline MPI::Datatype lookup(std::complex<double>*) { return MPI::DOUBLE_COMPLEX; }
    static inline MPI::Datatype lookup(std::complex<float>*) { return MPI::COMPLEX; }
    static inline MPI::Datatype lookup(std::complex<long double>*) { return MPI::LONG_DOUBLE_COMPLEX; }
    static inline MPI::Datatype lookup(unsigned*) { return MPI::UNSIGNED; }
    static inline MPI::Datatype lookup(unsigned char*) { return MPI::UNSIGNED_CHAR; }
    static inline MPI::Datatype lookup(unsigned long*) { return MPI::UNSIGNED_LONG; }
    static inline MPI::Datatype lookup(unsigned long long*) { return MPI::UNSIGNED_LONG_LONG; }
    static inline MPI::Datatype lookup(unsigned short*) { return MPI::UNSIGNED_SHORT; }
    static inline MPI::Datatype lookup(wchar_t*) { return MPI::WCHAR; }
    static inline MPI::Datatype lookup(Rim*) { return MPI::RIM; }
    static inline MPI::Datatype lookup(Tire*) { return MPI::TIRE; }
    static inline MPI::Datatype lookup(Wheel*) { return MPI::WHEEL; }
};

#endif
