#include <libgeodecomp/storage/unstructuredgrid.h>
#include <cxxtest/TestSuite.h>
#include <iostream>
#include <fstream>
#include <sstream>



class MyDummyElement
{
public:
    MyDummyElement(int const val=0):val_(val){}

    int const & operator() (int const & val) const{
        val_ = val;
        return val_;
    }

    int& operator() (int const & val){
        val_ = val;
        return val_;
    }

    int const & operator() ()const{
        return val_;
    }
    int& operator() (){
        return val_;
    }

private:
    int val_;

};

ostream& operator<< (ostream& out, MyDummyElement const & val){
    out << val();
    return out;
}


using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class UnstructuredGridTest : public CxxTest::TestSuite
{
    UnstructuredGrid<MyDummyElement> grid
public:
    void testFoo(){
    
    }

};

}
