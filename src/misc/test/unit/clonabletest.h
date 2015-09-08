#include <libgeodecomp/communication/hpxserializationwrapper.h>
#include <libgeodecomp/misc/clonable.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {


class Base
{
public:
    virtual ~Base()
    {}

    virtual Base *clone() const = 0;
    virtual int& value() = 0;
};

class Derived : public Clonable<Base, Derived>
{
public:
    explicit
    Derived(int v = 0) :
        v(v)
    {}

    int& value()
    {
        return v;
    }

private:
    int v;
};

class ClonableTest : public CxxTest::TestSuite
{
public:

    void testBasic()
    {
        Base *c = new Derived(5);
        Base *d = c->clone();

        TS_ASSERT_EQUALS(Derived().value(), 0);
        TS_ASSERT_EQUALS(c->value(), 5);
        TS_ASSERT_EQUALS(d->value(), 5);

        c->value() = 4711;
        TS_ASSERT_EQUALS(c->value(), 4711);
        TS_ASSERT_EQUALS(d->value(), 5);
    }

};

}
