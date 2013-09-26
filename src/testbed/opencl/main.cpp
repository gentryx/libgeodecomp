/* vim:set expandtab tabstop=2 shiftwidth=2 softtabstop=2: */

#include <iostream>
#include <libgeodecomp/libgeodecomp.h>

#include "openclstepper.h"

using namespace LibGeoDecomp;

#define MYCELL_STRUCT  \
    typedef struct {   \
      int x, y, z;      \
    } MyCell;

MYCELL_STRUCT

#define STRINGIFY(STRING) #STRING

class DummyCell : public OpenCLCellInterface<DummyCell, MyCell>
{
public:
    class API :
        public APITraits::HasStencil<Stencils::VonNeumann<3, 1> >,
        public APITraits::HasCubeTopology<3>
    {};

    DummyCell(int x, int y, int z)
    {
        myCellData.x = x;
        myCellData.y = y;
        myCellData.z = z;
    }

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, const unsigned& nanoStep)
    {}

    static std::string kernel_file()
    {
        return "./test.cl";
    }

    static std::string kernel_function()
    {
        return "add_test";
    }

    MyCell *data()
    {
        return &myCellData;
    }

    MyCell myCellData;
};

class DummyCellInitializer : public SimpleInitializer<DummyCell>
{
public:
    using SimpleInitializer<DummyCell>::gridDimensions;

    DummyCellInitializer() :
        SimpleInitializer<DummyCell>(Coord<3>(2, 2, 2))
    {}

    virtual void grid(GridBase<DummyCell, 3> *ret)
    {
        Coord<3> offset = Coord<3>(0, 0, 0);

        for (int z = 0; z < gridDimensions().z(); ++z) {
            for (int y = 0; y < gridDimensions().y(); ++y) {
                for (int x = 0; x < gridDimensions().x(); ++x) {
                    Coord<3> c = offset + Coord<3>(x, y, z);
                    ret->set(c, DummyCell(x, y, z));
                }
            }
        }
    }
};

class JacobiCell : public OpenCLCellInterface<JacobiCell, double>
{
public:
    class API :
        public APITraits::HasStencil<Stencils::VonNeumann<3, 1> >,
        public APITraits::HasCubeTopology<3>
    {};

    JacobiCell(double temp = 0) :
        temp(temp)
    {}

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, const unsigned& nanoStep)
    {}

    static std::string kernel_file()
    {
        return "./jacobi3d.cl";
    }

    static std::string kernel_function()
    {
        return "update";
    }

    double *data()
    {
        return &temp;
    }

    double temp;
};

class JacobiCellInitializer : public SimpleInitializer<JacobiCell>
{
public:
    JacobiCellInitializer(int size, int steps) :
        SimpleInitializer<JacobiCell>(Coord<3>::diagonal(size), steps)
    {}

    virtual void grid(GridBase<JacobiCell, 3> *ret)
    {
        Coord<3> offset = Coord<3>(0, 0, 0);

        for (int z = 0; z < dimensions.z(); ++z) {
            for (int y = 0; y < dimensions.y(); ++y) {
                for (int x = 0; x < dimensions.x(); ++x) {
                    Coord<3> c = offset + Coord<3>(x, y, z);
                    ret->set(c, JacobiCell(0.99999999999));
                }
            }
        }
    }
};

int main(int argc, char **argv)
{
    int size = 32;
    int steps = 100;

    typedef APITraits::SelectTopology<JacobiCell>::Value Topology;
    boost::shared_ptr<HiParSimulator::PartitionManager<Topology>> partitionManager(
        new HiParSimulator::PartitionManager<Topology>(
            CoordBox<3>(Coord<3>(0,0,0), Coord<3>::diagonal(size))));

    boost::shared_ptr<JacobiCellInitializer> initizalizer(
        new JacobiCellInitializer(size, steps));

    HiParSimulator::OpenCLStepper<JacobiCell, double> openclstepper(0, 0, partitionManager, initizalizer);

    openclstepper.update(2);

    return 0;
}
