#ifndef __cxxtest__TimingPrinter_h__
#define __cxxtest__TimingPrinter_h__

//
// The TimingPrinter is a simple TestListener that
// FIXME
//

#include <sys/time.h>
#include <time.h>
#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <cxxtest/Flags.h>

#ifndef _CXXTEST_HAVE_STD
#   define _CXXTEST_HAVE_STD
#endif // _CXXTEST_HAVE_STD

#include <cxxtest/ErrorFormatter.h>
#include <cxxtest/StdValueTraits.h>

#ifdef _CXXTEST_OLD_STD
#   include <iostream.h>
#else // !_CXXTEST_OLD_STD
#   include <iostream>
#endif // _CXXTEST_OLD_STD

namespace CxxTest 
{
    class TimingPrinter : public ErrorFormatter
    {
    public:
        TimingPrinter( CXXTEST_STD(ostream) &o = CXXTEST_STD(cout),
                const char *preLine = ":", const char *postLine = "" ) :
            ErrorFormatter( new Adapter(o), preLine, postLine ) 
        {
            CXXTEST_STD(cout) << CXXTEST_STD(setprecision(3)) << 
                CXXTEST_STD(fixed);
        }

        virtual ~TimingPrinter() { delete outputStream(); }

        void enterTest( const TestDescription & td)
        {
            ErrorFormatter::enterTest(td);
            CXXTEST_STD(cout) << td.suiteName() << "::" << td.testName() << ": ";
            _testtimer = now();
        }

        void leaveTest( const TestDescription & td)
        {
            _testtimer = now() - _testtimer;
            ErrorFormatter::leaveTest(td);
            CXXTEST_STD(cout) << " (" << _testtimer << "s)\n";
        }

        void enterSuite( const SuiteDescription & sd)
        {
            ErrorFormatter::enterSuite(sd);
            CXXTEST_STD(cout) << "\n";
            _suitetimer = now();
        }

        void leaveSuite( const SuiteDescription & sd)
        {
            _suitetimer = now() - _suitetimer;
            ErrorFormatter::leaveSuite(sd);
            CXXTEST_STD(cout) << sd.suiteName() << " total: " << _suitetimer <<
                "s\n";
        }

        void enterWorld( const WorldDescription & wd)
        {
            ErrorFormatter::enterWorld(wd);
            _worldtimer = now();
        }

        void leaveWorld( const WorldDescription & wd)
        {
            _worldtimer = now() - _worldtimer;
            CXXTEST_STD(cout) << "Total test time: " << _worldtimer << "s\n";
            ErrorFormatter::leaveWorld(wd);
        }

    private:
        double _testtimer;
        double _suitetimer;
        double _worldtimer;

        double now()
        {
            struct timeval tv;
            if (gettimeofday(&tv, NULL)) {
                perror("gettimeofday failed");
                abort();
            }
            return tv.tv_sec + 1.0e-6 * tv.tv_usec;
        }

        class Adapter : public OutputStream
        {
            CXXTEST_STD(ostream) &_o;
        public:
            Adapter( CXXTEST_STD(ostream) &o ) : _o(o) {}
            void flush() { _o.flush(); }
            OutputStream &operator<<( const char *s ) { _o << s; return *this; }
            OutputStream &operator<<( Manipulator m ) { return OutputStream::operator<<( m ); }
            OutputStream &operator<<( unsigned i )
            {
                char s[1 + 3 * sizeof(unsigned)];
                numberToString( i, s );
                _o << s;
                return *this;
            }
        };
    };
}

#endif // __cxxtest__TimingPrinter_h__
