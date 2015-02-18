#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/communication/mpilayer.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

class CellA
{
public:
    static MPI_Datatype MPIDataType;

    class API : public APITraits::HasCustomMPIDataType<CellA>
    {};

    CellA(double valA, char valB, int valC) :
        valA(valA),
        valB(valB),
        valC(valC)
    {}

    bool operator!=(const CellA& other) const
    {
        return !(*this == other);
    }

    bool operator==(const CellA& other) const
    {
        return
            (valA == other.valA) &&
            (valB == other.valB) &&
            (valC == other.valC);
    }

    double valA;
    char valB;
    int valC;
};

class CellB
{
public:
    class API : public APITraits::HasPredefinedMPIDataType<double>
    {};

    explicit CellB(double valA) :
        valA(valA)
    {}

    bool operator==(const CellB& other) const
    {
        return
            (valA == other.valA);
    }

    bool operator!=(const CellB& other) const
    {
        return !(*this == other);
    }


    double valA;
};

class CellC
{
public:
    class API : public APITraits::HasOpaqueMPIDataType<CellC>
    {};

    CellC(double valA, long valB, double valC) :
        valA(valA),
        valB(valB),
        valC(valC)
    {}

    bool operator==(const CellC& other) const
    {
        return
            (valA == other.valA) &&
            (valB == other.valB) &&
            (valC == other.valC);
    }

    bool operator!=(const CellC& other) const
    {
        return !(*this == other);
    }

    double valA;
    long valB;
    char valC;
};

class CellD
{
public:
    class API : public APITraits::HasOpaqueMPIDataType<CellD>
    {};

    CellD(char valA) :
        valA(valA)
    {}

    bool operator==(const CellD& other) const
    {
        return (valA == other.valA);
    }

    bool operator!=(const CellD& other) const
    {
        return !(*this == other);
    }

    char valA;
};

MPI_Datatype CellA::MPIDataType = MPI_DATATYPE_NULL;

class APITraitsTest : public CxxTest::TestSuite
{
public:

    void setUp()
    {
        int count = 3;
        CellA cell(1, 'a', 3);

        int blockLengths[] = {1, 1, 1};

        MPI_Aint base;
        MPI_Get_address(&cell, &base);
        MPI_Aint addresses[3];
        MPI_Get_address(&cell.valA, addresses + 0);
        MPI_Get_address(&cell.valB, addresses + 1);
        MPI_Get_address(&cell.valC, addresses + 2);

        MPI_Aint displacements[] = {
            addresses[0] - base,
            addresses[1] - base,
            addresses[2] - base};

        MPI_Datatype memberTypes[] = {
            MPI_DOUBLE,
            MPI_CHAR,
            MPI_INT
        };

        MPI_Type_create_struct(count, blockLengths, displacements, memberTypes, &CellA::MPIDataType);
        MPI_Type_commit(&CellA::MPIDataType);
    }

    void testMPIDatatypeSelectionCellA()
    {
        MPI_Datatype t1 = MPI_DATATYPE_NULL;
        MPI_Datatype t2 = MPI_DATATYPE_NULL;

        TS_ASSERT_EQUALS(t1, MPI_DATATYPE_NULL);
        TS_ASSERT_EQUALS(t2, MPI_DATATYPE_NULL);

        t1 = APITraits::SelectMPIDataType<CellA>::value();
        t2 = APITraits::SelectMPIDataType<CellA>::value();
        TS_ASSERT_EQUALS(t1, t2);
        TS_ASSERT_DIFFERS(t1, MPI_DATATYPE_NULL);

        CellA cell1(918273645, 'Z', 27644437);
        CellA cell2(0, 0, 0);
        TS_ASSERT_DIFFERS(cell1, cell2);

        MPILayer mpiLayer;
        mpiLayer.send(&cell1, 0, 1, 4713, t1);
        mpiLayer.recv(&cell2, 0, 1, 4713, t1);
        mpiLayer.wait(4713);
        TS_ASSERT_EQUALS(cell1, cell2);
    }

    void testMPIDatatypeSelectionCellB()
    {
        MPI_Datatype t1 = MPI_DATATYPE_NULL;
        MPI_Datatype t2 = MPI_DATATYPE_NULL;

        TS_ASSERT_EQUALS(t1, MPI_DATATYPE_NULL);
        TS_ASSERT_EQUALS(t2, MPI_DATATYPE_NULL);

        t1 = APITraits::SelectMPIDataType<CellB>::value();
        t2 = APITraits::SelectMPIDataType<CellB>::value();
        TS_ASSERT_EQUALS(t1, t2);
        TS_ASSERT_EQUALS(t1, MPI_DOUBLE);
        TS_ASSERT_DIFFERS(t1, MPI_DATATYPE_NULL);

        CellB cell1(69);
        CellB cell2(0);
        TS_ASSERT_DIFFERS(cell1, cell2);

        MPILayer mpiLayer;
        mpiLayer.send(&cell1, 0, 1, 4712, t1);
        mpiLayer.recv(&cell2, 0, 1, 4712, t1);
        mpiLayer.wait(4712);
        TS_ASSERT_EQUALS(cell1, cell2);
    }

    void testMPIDatatypeSelectionCellC()
    {
        MPI_Datatype t1 = MPI_DATATYPE_NULL;
        MPI_Datatype t2 = MPI_DATATYPE_NULL;

        TS_ASSERT_EQUALS(t1, MPI_DATATYPE_NULL);
        TS_ASSERT_EQUALS(t2, MPI_DATATYPE_NULL);

        t1 = APITraits::SelectMPIDataType<CellC>::value();
        t2 = APITraits::SelectMPIDataType<CellC>::value();
        TS_ASSERT_EQUALS(t1, t2);
        TS_ASSERT_DIFFERS(t1, MPI_DATATYPE_NULL);

        CellC cell1(1.23, 45678901234, 5.67);
        CellC cell2(0, 0, 0);
        TS_ASSERT_DIFFERS(cell1, cell2);

        MPILayer mpiLayer;
        mpiLayer.send(&cell1, 0, 1, 4711, t1);
        mpiLayer.recv(&cell2, 0, 1, 4711, t1);
        mpiLayer.wait(4711);
        TS_ASSERT_EQUALS(cell1, cell2);
    }

    void testMPIDatatypeCreationWithCellCD()
    {
        MPI_Datatype t1 = MPI_DATATYPE_NULL;
        MPI_Datatype t2 = MPI_DATATYPE_NULL;
        MPILayer mpiLayer;

        t1 = APITraits::SelectMPIDataType<CellC>::value();
        t2 = APITraits::SelectMPIDataType<CellD>::value();
        TS_ASSERT_DIFFERS(t1, t2);
        TS_ASSERT_DIFFERS(t1, MPI_DATATYPE_NULL);
        TS_ASSERT_DIFFERS(t2, MPI_DATATYPE_NULL);

        CellC cell1(1.23, 45678901234, 5.67);
        CellC cell2(0, 0, 0);
        TS_ASSERT_DIFFERS(cell1, cell2);

        mpiLayer.send(&cell1, 0, 1, 4711, t1);
        mpiLayer.recv(&cell2, 0, 1, 4711, t1);
        mpiLayer.wait(4711);
        TS_ASSERT_EQUALS(cell1, cell2);

        CellD cell3('4');
        CellD cell4('5');
        TS_ASSERT_DIFFERS(cell3, cell4);

        mpiLayer.send(&cell3, 0, 1, 69, t2);
        mpiLayer.recv(&cell4, 0, 1, 69, t2);
        mpiLayer.wait(69);
        TS_ASSERT_EQUALS(cell3, cell4);
    }
};

}
