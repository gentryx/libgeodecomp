#include <libgeodecomp/config.h>
#include <libgeodecomp/storage/sellcsigmasparsematrixcontainer.h>
#include <libgeodecomp/geometry/coord.h>
#include <cxxtest/TestSuite.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <map>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class SellCSigmaSparseMatrixContainerTest : public CxxTest::TestSuite
{
public:
    // test with a 8x8 diagonal Matrix, C = 1; Sigma = 1
    void testGetRow_one()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
//std::cout << "\n\n\nTEST 1: C=1 Sigma=1 Diagonalmatrix sotiert" <<std::endl;
        SellCSigmaSparseMatrixContainer<int, 1, 1> smc(8);

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

        smc.addPoint(0, 0, 1);
        smc.addPoint(1, 1, 2);
        smc.addPoint(2, 2, 3);
        smc.addPoint(3, 3, 4);
        smc.addPoint(4, 4, 5);
        smc.addPoint(5, 5, 6);
        smc.addPoint(6, 6, 7);
        smc.addPoint(7, 7, 8);

        std::vector< std::pair<int, int> > row0;
        std::pair<int, int> pair0(0, 1);
        row0.push_back(pair0);
        std::vector< std::pair<int, int> > row1;
        std::pair<int, int> pair1(1, 2);
        row1.push_back(pair1);
        std::vector< std::pair<int, int> > row2;
        std::pair<int, int> pair2(2, 3);
        row2.push_back(pair2);
        std::vector< std::pair<int, int> > row3;
        std::pair<int, int> pair3(3, 4);
        row3.push_back(pair3);
        std::vector< std::pair<int, int> > row4;
        std::pair<int, int> pair4(4, 5);
        row4.push_back(pair4);
        std::vector< std::pair<int, int> > row5;
        std::pair<int, int> pair5(5, 6);
        row5.push_back(pair5);
        std::vector< std::pair<int, int> > row6;
        std::pair<int, int> pair6(6, 7);
        row6.push_back(pair6);
        std::vector< std::pair<int, int> > row7;
        std::pair<int, int> pair7(7, 8);
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
#endif
    }

    // test with a 8x8 diagonal Matrix, C = 1; Sigma = 1
    // randome addPoints
    void testGetRow_two()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        SellCSigmaSparseMatrixContainer<int, 1, 1> smc(8);

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

        smc.addPoint(1, 1, 2);
        smc.addPoint(0, 0, 1);
        smc.addPoint(6, 6, 7);
        smc.addPoint(4, 4, 5);
        smc.addPoint(7, 7, 8);
        smc.addPoint(2, 2, 3);
        smc.addPoint(5, 5, 6);
        smc.addPoint(3, 3, 4);

        std::vector< std::pair<int, int> > row0;
        std::pair<int, int> pair0(0, 1);
        row0.push_back(pair0);
        std::vector< std::pair<int, int> > row1;
        std::pair<int, int> pair1(1, 2);
        row1.push_back(pair1);
        std::vector< std::pair<int, int> > row2;
        std::pair<int, int> pair2(2, 3);
        row2.push_back(pair2);
        std::vector< std::pair<int, int> > row3;
        std::pair<int, int> pair3(3, 4);
        row3.push_back(pair3);
        std::vector< std::pair<int, int> > row4;
        std::pair<int, int> pair4(4, 5);
        row4.push_back(pair4);
        std::vector< std::pair<int, int> > row5;
        std::pair<int, int> pair5(5, 6);
        row5.push_back(pair5);
        std::vector< std::pair<int, int> > row6;
        std::pair<int, int> pair6(6, 7);
        row6.push_back(pair6);
        std::vector< std::pair<int, int> > row7;
        std::pair<int, int> pair7(7, 8);
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
#endif
    }

    // test with a 8x8 diagonal Matrix, C = 2; Sigma = 1
    // randome addPoints
    void testGetRow_three()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        int const C(2);
        int const SIGMA(1);
        SellCSigmaSparseMatrixContainer<int, C, SIGMA> smc(8);

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

        smc.addPoint(5, 5, 6);
        smc.addPoint(1, 1, 2);
        smc.addPoint(4, 4, 5);
        smc.addPoint(2, 2, 3);
        smc.addPoint(0, 0, 1);
        smc.addPoint(6, 6, 7);
        smc.addPoint(7, 7, 8);
        smc.addPoint(3, 3, 4);

        std::vector< std::pair<int, int> > row0;
        std::pair<int, int> pair0(0, 1);
        row0.push_back(pair0);
        std::vector< std::pair<int, int> > row1;
        std::pair<int, int> pair1(1, 2);
        row1.push_back(pair1);
        std::vector< std::pair<int, int> > row2;
        std::pair<int, int> pair2(2, 3);
        row2.push_back(pair2);
        std::vector< std::pair<int, int> > row3;
        std::pair<int, int> pair3(3, 4);
        row3.push_back(pair3);
        std::vector< std::pair<int, int> > row4;
        std::pair<int, int> pair4(4, 5);
        row4.push_back(pair4);
        std::vector< std::pair<int, int> > row5;
        std::pair<int, int> pair5(5, 6);
        row5.push_back(pair5);
        std::vector< std::pair<int, int> > row6;
        std::pair<int, int> pair6(6, 7);
        row6.push_back(pair6);
        std::vector< std::pair<int, int> > row7;
        std::pair<int, int> pair7(7, 8);
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
#endif
    }

    // test with a 8x8 diagonal Matrix, C = 2; Sigma = 1
    void testGetRow_fore()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        int const C(3);
        int const SIGMA(1);
        SellCSigmaSparseMatrixContainer<int, C, SIGMA> smc(8);

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

        smc.addPoint(0, 0, 1);
        smc.addPoint(1, 1, 2);
        smc.addPoint(2, 2, 3);
        smc.addPoint(3, 3, 4);
        smc.addPoint(4, 4, 5);
        smc.addPoint(5, 5, 6);
        smc.addPoint(6, 6, 7);
        smc.addPoint(7, 7, 8);

        std::vector< std::pair<int, int> > row0;
        std::pair<int, int> pair0(0, 1);
        row0.push_back(pair0);
        std::vector< std::pair<int, int> > row1;
        std::pair<int, int> pair1(1, 2);
        row1.push_back(pair1);
        std::vector< std::pair<int, int> > row2;
        std::pair<int, int> pair2(2, 3);
        row2.push_back(pair2);
        std::vector< std::pair<int, int> > row3;
        std::pair<int, int> pair3(3, 4);
        row3.push_back(pair3);
        std::vector< std::pair<int, int> > row4;
        std::pair<int, int> pair4(4, 5);
        row4.push_back(pair4);
        std::vector< std::pair<int, int> > row5;
        std::pair<int, int> pair5(5, 6);
        row5.push_back(pair5);
        std::vector< std::pair<int, int> > row6;
        std::pair<int, int> pair6(6, 7);
        row6.push_back(pair6);
        std::vector< std::pair<int, int> > row7;
        std::pair<int, int> pair7(7, 8);
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
#endif
    }

    // test with a 9x9 sparse Matrix, C = 3; Sigma = 1
    void testGetRow_five()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        int const C(3);
        int const SIGMA(1);
        SellCSigmaSparseMatrixContainer<char, C, SIGMA> smc(9);

        /* add a test 8x8 Matrix:
         *              col
         * row | 0 1 2 3 4 5 6 7 8     chunk #
         * ____|_______________________________
         *  0  | 0 0 0 0 A 0 0 0 0 |
         *  1  | 0 0 0 0 0 0 0 0 0 |-> chunk 0
         *  2  | 0 0 0 0 B 0 C 0 0 |
         * ----+-------------------------------
         *  3  | 0 0 D 0 0 0 0 E F |
         *  4  | G 0 0 0 0 H 0 0 0 |-> chunk 1
         *  5  | 0 I J 0 0 0 0 0 0 |
         * ----+-------------------------------
         *  6  | 0 0 0 K 0 0 0 0 0 |
         *  7  | L 0 M 0 N 0 O 0 P |-> chunk 2
         *  8  | 0 0 0 S 0 Q 0 R 0 |
         */

        smc.addPoint(4, 0, 'G');
        smc.addPoint(8, 7, 'R');
        smc.addPoint(6, 3, 'K');
        smc.addPoint(7, 6, 'O');
        smc.addPoint(7, 4, 'N');
        smc.addPoint(7, 2, 'M');
        smc.addPoint(0, 4, 'A');
        smc.addPoint(3, 8, 'F');
        smc.addPoint(7, 0, 'L');
        smc.addPoint(3, 7, 'E');
        smc.addPoint(2, 4, 'B');
        smc.addPoint(5, 1, 'I');
        smc.addPoint(7, 8, 'P');
        smc.addPoint(3, 2, 'D');
        smc.addPoint(8, 5, 'Q');
        smc.addPoint(8, 3, 'S');
        smc.addPoint(5, 2, 'J');
        smc.addPoint(4, 5, 'H');
        smc.addPoint(2, 6, 'C');

        std::vector< std::pair<int, char> > row0;
        std::pair<int, char> pair0(4, 'A');
        row0.push_back(pair0);

        std::vector< std::pair<int, char> > row1;

        std::vector< std::pair<int, char> > row2;
        std::pair<int, char> pair1(4, 'B');
        std::pair<int, char> pair2(6, 'C');
        row2.push_back(pair1);
        row2.push_back(pair2);

        std::vector< std::pair<int, char> > row3;
        std::pair<int, char> pair3(2, 'D');
        std::pair<int, char> pair4(7, 'E');
        std::pair<int, char> pair5(8, 'F');
        row3.push_back(pair3);
        row3.push_back(pair4);
        row3.push_back(pair5);

        std::vector< std::pair<int, char> > row4;
        std::pair<int, char> pair6(0, 'G');
        std::pair<int, char> pair7(5, 'H');
        row4.push_back(pair6);
        row4.push_back(pair7);

        std::vector< std::pair<int, char> > row5;
        std::pair<int, char> pair8(1, 'I');
        std::pair<int, char> pair9(2, 'J');
        row5.push_back(pair8);
        row5.push_back(pair9);

        std::vector< std::pair<int, char> > row6;
        std::pair<int, char> pair10(3, 'K');
        row6.push_back(pair10);

        std::vector< std::pair<int, char> > row7;
        std::pair<int, char> pair11(0, 'L');
        std::pair<int, char> pair12(2, 'M');
        std::pair<int, char> pair13(4, 'N');
        std::pair<int, char> pair14(6, 'O');
        std::pair<int, char> pair15(8, 'P');
        row7.push_back(pair11);
        row7.push_back(pair12);
        row7.push_back(pair13);
        row7.push_back(pair14);
        row7.push_back(pair15);

        std::vector< std::pair<int, char> > row8;
        std::pair<int, char> pair16(5, 'Q');
        std::pair<int, char> pair17(7, 'R');
        std::pair<int, char> pair18(3, 'S');
        row8.push_back(pair18);
        row8.push_back(pair16);
        row8.push_back(pair17);


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
        TS_ASSERT_EQUALS(
                smc.getRow(8),
                row8
                );
#endif
    }

    // test with a 3x3 diagonal Matrix, C = 2; Sigma = 1
    // overwite a value
    void testGetRow_six()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        int const C(2);
        int const SIGMA(1);
        SellCSigmaSparseMatrixContainer<int, C, SIGMA> smc(3);

        /* add a test 3x3 Matrix:
         * 1 0 4
         * 0 5 0
         * 0 0 6
         */

        smc.addPoint(0, 0, 1);
        smc.addPoint(0, 2, 4);
        smc.addPoint(1, 1, 2);
        smc.addPoint(2, 2, 3);

        smc.addPoint(1, 1, 5);
        smc.addPoint(2, 2, 6);

        std::vector< std::pair<int, int> > row0;
        std::pair<int, int> pair0(0, 1);
        std::pair<int, int> pair3(2, 4);
        row0.push_back(pair0);
        row0.push_back(pair3);
        std::vector< std::pair<int, int> > row1;
        std::pair<int, int> pair1(1, 5);
        row1.push_back(pair1);
        std::vector< std::pair<int, int> > row2;
        std::pair<int, int> pair2(2, 6);
        row2.push_back(pair2);

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
#endif
    }

    void testMatVecMul_diag()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        SellCSigmaSparseMatrixContainer<int, 4, 1> smc(8);

        /* add a test 8x8 Matrix:
         * 1       1 0 0 0 0 0 0 0      1
         * 2       0 1 0 0 0 0 0 0      2
         * 3       0 0 1 0 0 0 0 0      3
         * 4       0 0 0 1 0 0 0 0      4
         * 5   =   0 0 0 0 1 0 0 0  x   5
         * 6       0 0 0 0 0 1 0 0      6
         * 7       0 0 0 0 0 0 1 0      7
         * 8       0 0 0 0 0 0 0 1      8
         */

        smc.addPoint(0, 0, 1);
        smc.addPoint(1, 1, 1);
        smc.addPoint(2, 2, 1);
        smc.addPoint(3, 3, 1);
        smc.addPoint(4, 4, 1);
        smc.addPoint(5, 5, 1);
        smc.addPoint(6, 6, 1);
        smc.addPoint(7, 7, 1);

        std::vector<int> lhs(8);
        std::vector<int> rhs(8);
        std::vector<int> expected(8);

        for (int i=0; i<8; ++i) {
            rhs[i]      = i+1;
            expected[i] = i+1;
        }

        smc.matVecMul(lhs, rhs);

        TS_ASSERT_EQUALS(
                lhs,
                expected
                );
#endif
    }

    void testMatVecMul_easy()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        SellCSigmaSparseMatrixContainer<int, 2, 1> smc(4);

        /* add a test 4x4 Matrix:
         * 3       0 0 1 0       1
         * 1       1 0 0 0       2
         * 4   =   0 0 0 1   x   3
         * 2       0 1 0 0       4
         */

        smc.addPoint(0, 2, 1);
        smc.addPoint(1, 0, 1);
        smc.addPoint(2, 3, 1);
        smc.addPoint(3, 1, 1);

        std::vector<int> lhs(4);
        std::vector<int> rhs(4);
        std::vector<int> expected(4);

        for (int i=0; i<4; ++i) {
            rhs[i] = i+1;
        }

        expected[0] = 3;
        expected[1] = 1;
        expected[2] = 4;
        expected[3] = 2;

        smc.matVecMul(lhs, rhs);

        TS_ASSERT_EQUALS(
                lhs,
                expected
                );
#endif
    }


    void testMatVecMul_two()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        SellCSigmaSparseMatrixContainer<int, 2, 1> smc(4);

        /* add a test 4x4 Matrix:
         * -4       0 1 2 -3       1
         * 12       -4 3 2 1       2
         * 12   =   2 -3 4 1   x   3
         * 13       3 4 2 -1       4
         */

        smc.addPoint(0, 0, 0);
        smc.addPoint(0, 1, 1);
        smc.addPoint(0, 2, 2);
        smc.addPoint(0, 3, -3);
        smc.addPoint(1, 0, -4);
        smc.addPoint(1, 1, 3);
        smc.addPoint(1, 2, 2);
        smc.addPoint(1, 3, 1);
        smc.addPoint(2, 0, 2);
        smc.addPoint(2, 1, -3);
        smc.addPoint(2, 2, 4);
        smc.addPoint(2, 3, 1);
        smc.addPoint(3, 0, 3);
        smc.addPoint(3, 1, 4);
        smc.addPoint(3, 2, 2);
        smc.addPoint(3, 3, -1);

        std::vector<int> lhs(4);
        std::vector<int> rhs(4);
        std::vector<int> expected(4);

        for (int i=0; i<4; ++i) {
            rhs[i]      = i+1;
        }

        expected[0] = -4;
        expected[1] = 12;
        expected[2] = 12;
        expected[3] = 13;

        smc.matVecMul(lhs, rhs);

        TS_ASSERT_EQUALS(
                lhs,
                expected
                );
#endif
    }

    void testEqualOperator()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        const int DIM = 128;
        SellCSigmaSparseMatrixContainer<double,2,1> a(DIM);
        SellCSigmaSparseMatrixContainer<double,2,1> b(DIM);

        for (int i=0; i<DIM; ++i) {
            a.addPoint(i, abs(i*17+57)%DIM, (i*17+57)/DIM);
            a.addPoint(i, abs(i*17-57)%DIM, (i*17-57)/DIM);
            a.addPoint(i, abs(i*57+17)%DIM, (i*57+17)/DIM);

            b.addPoint(i, abs(i*17+57)%DIM, (i*17+57)/DIM);
            b.addPoint(i, abs(i*17-57)%DIM, (i*17-57)/DIM);
            b.addPoint(i, abs(i*57+17)%DIM, (i*57+17)/DIM);
        }

        TS_ASSERT(a == a);
        TS_ASSERT(a == b);

        b.addPoint(DIM-17, DIM - 7, 0.666);

        TS_ASSERT_EQUALS((a == b), false);
#endif
    }

    void testInitFromMatrix()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        SellCSigmaSparseMatrixContainer<double, 2, 1> a(4);
        std::map<Coord<2>, double> matrix;
        // 0 1 0 2
        // 0 0 0 1
        // 0 2 0 3
        // 1 1 0 0
        matrix[Coord<2>(0, 1)] = 1;
        matrix[Coord<2>(0, 3)] = 2;
        matrix[Coord<2>(1, 3)] = 1;
        matrix[Coord<2>(2, 1)] = 2;
        matrix[Coord<2>(2, 3)] = 3;
        matrix[Coord<2>(3, 0)] = 1;
        matrix[Coord<2>(3, 1)] = 1;

        a.initFromMatrix(4, matrix);

        const std::vector<double>& val = a.valuesVec();
        const std::vector<int>&    col = a.columnVec();
        const std::vector<int>&    cl  = a.chunkLengthVec();
        const std::vector<int>&    cs  = a.chunkOffsetVec();

        TS_ASSERT_EQUALS(2, std::distance(begin(cs),  end(cs)));
        TS_ASSERT_EQUALS(2, std::distance(begin(cl),  end(cl)));
        TS_ASSERT_EQUALS(8, std::distance(begin(val), end(val)));
        TS_ASSERT_EQUALS(8, std::distance(begin(col), end(col)));

        TS_ASSERT(cs[0] == 0);
        TS_ASSERT(cs[1] == 4);
        TS_ASSERT(cl[0] == 2);
        TS_ASSERT(cl[1] == 2);

        TS_ASSERT(val[0] == 1);
        TS_ASSERT(val[1] == 1);
        TS_ASSERT(val[2] == 2);
        TS_ASSERT(val[3] == 0);
        TS_ASSERT(val[4] == 2);
        TS_ASSERT(val[5] == 1);
        TS_ASSERT(val[6] == 3);
        TS_ASSERT(val[7] == 1);

        TS_ASSERT(col[0] == 1);
        TS_ASSERT(col[1] == 3);
        TS_ASSERT(col[2] == 3);
        TS_ASSERT(col[3] == 0);
        TS_ASSERT(col[4] == 1);
        TS_ASSERT(col[5] == 0);
        TS_ASSERT(col[6] == 3);
        TS_ASSERT(col[7] == 1);
#endif
    }

    void testInitFromMatrixWithoutSIGMA()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        SellCSigmaSparseMatrixContainer<double, 2, 1> a(5);
        std::map<Coord<2>, double> matrix;
        // 0 1 0 0 0
        // 0 5 0 8 7
        // 1 4 0 3 2
        // 0 0 5 0 0
        // 0 0 0 2 3
        matrix[Coord<2>(0, 1)] = 1;
        matrix[Coord<2>(1, 1)] = 5;
        matrix[Coord<2>(1, 3)] = 8;
        matrix[Coord<2>(1, 4)] = 7;
        matrix[Coord<2>(2, 0)] = 1;
        matrix[Coord<2>(2, 1)] = 4;
        matrix[Coord<2>(2, 3)] = 3;
        matrix[Coord<2>(2, 4)] = 2;
        matrix[Coord<2>(3, 2)] = 5;
        matrix[Coord<2>(4, 3)] = 2;
        matrix[Coord<2>(4, 4)] = 3;

        a.initFromMatrix(5, matrix);

        const std::vector<double>& val = a.valuesVec();
        const std::vector<int>&    col = a.columnVec();
        const std::vector<int>&    cl  = a.chunkLengthVec();
        const std::vector<int>&    cs  = a.chunkOffsetVec();

        TS_ASSERT_EQUALS(3,  std::distance(begin(cs),  end(cs)));
        TS_ASSERT_EQUALS(3,  std::distance(begin(cl),  end(cl)));
        TS_ASSERT_EQUALS(18, std::distance(begin(val), end(val)));
        TS_ASSERT_EQUALS(18, std::distance(begin(col), end(col)));

        TS_ASSERT(cs[0] ==  0);
        TS_ASSERT(cs[1] ==  6);
        TS_ASSERT(cs[2] == 14);
        TS_ASSERT(cl[0] == 3);
        TS_ASSERT(cl[1] == 4);
        TS_ASSERT(cl[2] == 2);

        TS_ASSERT(val[ 0] == 1);
        TS_ASSERT(val[ 1] == 5);
        TS_ASSERT(val[ 2] == 0);
        TS_ASSERT(val[ 3] == 8);
        TS_ASSERT(val[ 4] == 0);
        TS_ASSERT(val[ 5] == 7);
        TS_ASSERT(val[ 6] == 1);
        TS_ASSERT(val[ 7] == 5);
        TS_ASSERT(val[ 8] == 4);
        TS_ASSERT(val[ 9] == 0);
        TS_ASSERT(val[10] == 3);
        TS_ASSERT(val[11] == 0);
        TS_ASSERT(val[12] == 2);
        TS_ASSERT(val[13] == 0);
        TS_ASSERT(val[14] == 2);
        TS_ASSERT(val[15] == 0);
        TS_ASSERT(val[16] == 3);
        TS_ASSERT(val[17] == 0);

        TS_ASSERT(col[ 0] == 1);
        TS_ASSERT(col[ 1] == 1);
        TS_ASSERT(col[ 2] == 0);
        TS_ASSERT(col[ 3] == 3);
        TS_ASSERT(col[ 4] == 0);
        TS_ASSERT(col[ 5] == 4);
        TS_ASSERT(col[ 6] == 0);
        TS_ASSERT(col[ 7] == 2);
        TS_ASSERT(col[ 8] == 1);
        TS_ASSERT(col[ 9] == 0);
        TS_ASSERT(col[10] == 3);
        TS_ASSERT(col[11] == 0);
        TS_ASSERT(col[12] == 4);
        TS_ASSERT(col[13] == 0);
        TS_ASSERT(col[14] == 3);
        TS_ASSERT(col[15] == 0);
        TS_ASSERT(col[16] == 4);
        TS_ASSERT(col[17] == 0);
#endif
    }

    void testInitFromMatrixWithSIGMA1()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        SellCSigmaSparseMatrixContainer<double, 2, 4> a(5);
        std::map<Coord<2>, double> matrix;
        // 0 1 0 0 0
        // 0 5 0 8 7
        // 1 4 0 3 2
        // 0 0 5 0 0
        // 0 0 0 2 3
        matrix[Coord<2>(0, 1)] = 1;
        matrix[Coord<2>(1, 1)] = 5;
        matrix[Coord<2>(1, 3)] = 8;
        matrix[Coord<2>(1, 4)] = 7;
        matrix[Coord<2>(2, 0)] = 1;
        matrix[Coord<2>(2, 1)] = 4;
        matrix[Coord<2>(2, 3)] = 3;
        matrix[Coord<2>(2, 4)] = 2;
        matrix[Coord<2>(3, 2)] = 5;
        matrix[Coord<2>(4, 3)] = 2;
        matrix[Coord<2>(4, 4)] = 3;

        a.initFromMatrix(5, matrix);

        const std::vector<double>& val = a.valuesVec();
        const std::vector<int>&    col = a.columnVec();
        const std::vector<int>&    cl  = a.chunkLengthVec();
        const std::vector<int>&    cs  = a.chunkOffsetVec();

        TS_ASSERT_EQUALS(3,  std::distance(begin(cs),  end(cs)));
        TS_ASSERT_EQUALS(3,  std::distance(begin(cl),  end(cl)));
        TS_ASSERT_EQUALS(14, std::distance(begin(val), end(val)));
        TS_ASSERT_EQUALS(14, std::distance(begin(col), end(col)));

        TS_ASSERT(cs[0] ==  0);
        TS_ASSERT(cs[1] ==  8);
        TS_ASSERT(cs[2] == 10);
        TS_ASSERT(cl[0] == 4);
        TS_ASSERT(cl[1] == 1);
        TS_ASSERT(cl[2] == 2);

        TS_ASSERT(val[ 0] == 1);
        TS_ASSERT(val[ 1] == 5);
        TS_ASSERT(val[ 2] == 4);
        TS_ASSERT(val[ 3] == 8);
        TS_ASSERT(val[ 4] == 3);
        TS_ASSERT(val[ 5] == 7);
        TS_ASSERT(val[ 6] == 2);
        TS_ASSERT(val[ 7] == 0);
        TS_ASSERT(val[ 8] == 1);
        TS_ASSERT(val[ 9] == 5);
        TS_ASSERT(val[10] == 2);
        TS_ASSERT(val[11] == 0);
        TS_ASSERT(val[12] == 3);
        TS_ASSERT(val[13] == 0);

        TS_ASSERT(col[ 0] == 0);
        TS_ASSERT(col[ 1] == 1);
        TS_ASSERT(col[ 2] == 1);
        TS_ASSERT(col[ 3] == 3);
        TS_ASSERT(col[ 4] == 3);
        TS_ASSERT(col[ 5] == 4);
        TS_ASSERT(col[ 6] == 4);
        TS_ASSERT(col[ 7] == 0);
        TS_ASSERT(col[ 8] == 1);
        TS_ASSERT(col[ 9] == 2);
        TS_ASSERT(col[10] == 3);
        TS_ASSERT(col[11] == 0);
        TS_ASSERT(col[12] == 4);
        TS_ASSERT(col[13] == 0);
#endif
    }

    void testInitFromMatrixWithSIGMA2()
    {
#ifdef LIBGEODECOMP_WITH_CPP14
        SellCSigmaSparseMatrixContainer<double, 2, 8> a(5);
        std::map<Coord<2>, double> matrix;
        // 0 1 0 0 0
        // 0 5 0 8 7
        // 1 4 0 3 2
        // 0 0 5 0 0
        // 0 0 0 2 3
        matrix[Coord<2>(0, 1)] = 1;
        matrix[Coord<2>(1, 1)] = 5;
        matrix[Coord<2>(1, 3)] = 8;
        matrix[Coord<2>(1, 4)] = 7;
        matrix[Coord<2>(2, 0)] = 1;
        matrix[Coord<2>(2, 1)] = 4;
        matrix[Coord<2>(2, 3)] = 3;
        matrix[Coord<2>(2, 4)] = 2;
        matrix[Coord<2>(3, 2)] = 5;
        matrix[Coord<2>(4, 3)] = 2;
        matrix[Coord<2>(4, 4)] = 3;

        a.initFromMatrix(5, matrix);

        const std::vector<double>& val = a.valuesVec();
        const std::vector<int>&    col = a.columnVec();
        const std::vector<int>&    cl  = a.chunkLengthVec();
        const std::vector<int>&    cs  = a.chunkOffsetVec();

        TS_ASSERT_EQUALS(3,  std::distance(begin(cs),  end(cs)));
        TS_ASSERT_EQUALS(3,  std::distance(begin(cl),  end(cl)));
        TS_ASSERT_EQUALS(14, std::distance(begin(val), end(val)));
        TS_ASSERT_EQUALS(14, std::distance(begin(col), end(col)));

        TS_ASSERT(cs[0] ==  0);
        TS_ASSERT(cs[1] ==  8);
        TS_ASSERT(cs[2] == 12);
        TS_ASSERT(cl[0] == 4);
        TS_ASSERT(cl[1] == 2);
        TS_ASSERT(cl[2] == 1);

        TS_ASSERT(val[ 0] == 1);
        TS_ASSERT(val[ 1] == 5);
        TS_ASSERT(val[ 2] == 4);
        TS_ASSERT(val[ 3] == 8);
        TS_ASSERT(val[ 4] == 3);
        TS_ASSERT(val[ 5] == 7);
        TS_ASSERT(val[ 6] == 2);
        TS_ASSERT(val[ 7] == 0);
        TS_ASSERT(val[ 8] == 2);
        TS_ASSERT(val[ 9] == 1);
        TS_ASSERT(val[10] == 3);
        TS_ASSERT(val[11] == 0);
        TS_ASSERT(val[12] == 5);
        TS_ASSERT(val[13] == 0);

        TS_ASSERT(col[ 0] == 0);
        TS_ASSERT(col[ 1] == 1);
        TS_ASSERT(col[ 2] == 1);
        TS_ASSERT(col[ 3] == 3);
        TS_ASSERT(col[ 4] == 3);
        TS_ASSERT(col[ 5] == 4);
        TS_ASSERT(col[ 6] == 4);
        TS_ASSERT(col[ 7] == 0);
        TS_ASSERT(col[ 8] == 3);
        TS_ASSERT(col[ 9] == 1);
        TS_ASSERT(col[10] == 4);
        TS_ASSERT(col[11] == 0);
        TS_ASSERT(col[12] == 2);
        TS_ASSERT(col[13] == 0);
#endif
    }
};

}
