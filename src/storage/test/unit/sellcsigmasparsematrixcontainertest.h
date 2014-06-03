#include <libgeodecomp/storage/sellcsigmasparsematrixcontainer.h>
#include <cxxtest/TestSuite.h>
#include <iostream>


using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class SellCSigmaSparseMatrixContainerTest : public CxxTest::TestSuite
{
public:
    // test with a 8x8 diagonal Matrix, C = 1; Sigma = 1
    void testGetRow_one()
    {
        std::cout << "fooooooooooooooo" << std::endl;
        SellCSigmaSparseMatrixContainer<int, 1, 1> smc;

        /* add a test 8x8 Matrix:
         * 1 0 0 0 0 0 0 0
         * 0 2 0 0 0 0 0 0
         * 0 0 3 0 0 0 0 0
         * 0 0 0 4 0 0 0 0
         * 0 0 0 0 5 0 0 0
         * 0 0 0 0 0 6 0 0
         * 0 0 0 0 0 0 7 0
         * 0 0 0 0 0 0 0 8
         */

        smc.addPoint (0, 0, 1);
        smc.addPoint (1, 1, 2);
        smc.addPoint (2, 2, 3);
        smc.addPoint (3, 3, 4);
        smc.addPoint (4, 4, 5);
        smc.addPoint (5, 5, 6);
        smc.addPoint (6, 6, 7);
        smc.addPoint (7, 7, 8);

        std::vector< std::pair<int, int> > row0;
        std::pair<int, int> pair0 (0,1);
        row0.push_back(pair0);
        std::vector< std::pair<int, int> > row1;
        std::pair<int, int> pair1 (1,2);
        row1.push_back(pair1);
        std::vector< std::pair<int, int> > row2;
        std::pair<int, int> pair2 (2,3);
        row2.push_back(pair2);
        std::vector< std::pair<int, int> > row3;
        std::pair<int, int> pair3 (3,4);
        row3.push_back(pair3);
        std::vector< std::pair<int, int> > row4;
        std::pair<int, int> pair4 (4,5);
        row4.push_back(pair4);
        std::vector< std::pair<int, int> > row5;
        std::pair<int, int> pair5 (5,6);
        row5.push_back(pair5);
        std::vector< std::pair<int, int> > row6;
        std::pair<int, int> pair6 (6,7);
        row6.push_back(pair6);
        std::vector< std::pair<int, int> > row7;
        std::pair<int, int> pair7 (7,8);
        row7.push_back(pair7);


        TS_ASSERT_EQUALS(
                smc.getRow(0),
                row0
                );
        TS_ASSERT_EQUALS(
                smc.getRow(1),
                row1
                );
        TS_ASSERT_EQUALS(
                smc.getRow(2),
                row2
                );
        TS_ASSERT_EQUALS(
                smc.getRow(3),
                row3
                );
        TS_ASSERT_EQUALS(
                smc.getRow(4),
                row4
                );
        TS_ASSERT_EQUALS(
                smc.getRow(5),
                row5
                );
        TS_ASSERT_EQUALS(
                smc.getRow(6),
                row6
                );
        TS_ASSERT_EQUALS(
                smc.getRow(7),
                row7
                );
    }

    // test with a 4x4 Matrix, C = 1; Sigma = 1
/*    void testGetRow_one()
    {
        SellCSigmaSparseMatrixContainer<int, 1, 1> SMC();

         * add a test 8x8 Matrix:
         *  | 0 1 2 3
         *--+----------
         * 0| 1 2 3 0 
         * 1| 0 0 0 0 
         * 2| 0 4 5 0 
         * 3| 6 7 8 9 
         *

        SMC.addPoint (0, 0, 1);
        SMC.addPoint (0, 1, 2);
        SMC.addPoint (0, 2, 3);
        SMC.addPoint (2, 1, 4);
        SMC.addPoint (2, 2, 5);
        SMC.addPoint (3, 0, 6);
        SMC.addPoint (3, 1, 7);
        SMC.addPoint (3, 2, 8);
        SMC.addPoint (3, 3, 9);

        TS_ASSERT_EQUALS(
                SMC.getRow(0),
                std::vector< std::pair<int, int> > ( std::pair<int, int>(0,1) )
                );
        TS_ASSERT_EQUALS(
                SMC.getRow(1),
                std::vector< std::pair<int, int> > ( )
                );
        TS_ASSERT_EQUALS(
                SMC.getRow(2),
                std::vector< std::pair<int, int> > ( std::pair<int, int>(2,3) )
                );
        TS_ASSERT_EQUALS(
                SMC.getRow(3),
                std::vector< std::pair<int, int> > ( std::pair<int, int>(3,4) )
                );
    }
*/




;

}
