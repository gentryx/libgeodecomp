/* vim:set expandtab tabstop=2 shiftwidth=2 softtabstop=2: */

#include <fstream>
#include <iostream>
#include <libgeodecomp/libgeodecomp.h>
#include <libgeodecomp/misc/testcell.h>
#include <libgeodecomp/io/testinitializer.h>

#include "openclstepper.h"

using namespace LibGeoDecomp;

class DummyCell : public OpenCLCellInterface<DummyCell> {
  public:
    static const int DIMENSIONS = 3;
    typedef Stencils::VonNeumann<3, 1> Stencil;
    class API : public CellAPITraits::Base {};
    typedef Topologies::Cube<3>::Topology Topology;

    static inline unsigned nanoSteps() { return 1; }
    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, const unsigned& nanoStep) {}

#define MYCELL_STRUCT  \
    typedef struct {   \
      int x, y, z;      \
    } MyCell;

#define STRINGIFY(STRING) #STRING

    MYCELL_STRUCT

    static std::string kernel_file() { return "./test.cl"; }
    static std::string kernel_function() { return "dummy_test"; }
    static std::string cl_struct_code() { return STRINGIFY(MYCELL_STRUCT); }
    static size_t sizeof_data() { return sizeof(MyCell); }
    void * data() { return &myCellData; }

    MyCell myCellData;
};

class DummyCellInitializer : public SimpleInitializer<DummyCell> {
  public:
    using SimpleInitializer<DummyCell>::gridDimensions;

    DummyCellInitializer(void) : SimpleInitializer<DummyCell>(Coord<3>(2, 2, 2))
    {}

    virtual void grid(GridBase<DummyCell, 3> *ret)
    {
        Coord<3> offset = Coord<3>(0, 0, 0);

        for (int z = 0; z < gridDimensions().z(); ++z) {
          for (int y = 0; y < gridDimensions().y(); ++y) {
            for (int x = 0; x < gridDimensions().x(); ++x) {
              Coord<3> c = offset + Coord<3>(x, y, z);
                        ret->at(c) = DummyCell();
                        ret->at(c).myCellData.x = x;
                        ret->at(c).myCellData.y = y;
                        ret->at(c).myCellData.z = z;
                }
            }
        }
    }
};

int main(int argc, char **argv)
{
    typedef DummyCell MyCell;
    typedef DummyCellInitializer MyCellInitializer;

    boost::shared_ptr<HiParSimulator::PartitionManager<3>>
      pmp(new HiParSimulator::PartitionManager<3>(
            CoordBox<3>(Coord<3>(0,0,0), Coord<3>(2,2,2))));

    boost::shared_ptr<MyCellInitializer> dcip(new MyCellInitializer);

    HiParSimulator::OpenCLStepper<MyCell> openclstepper(0, 0, pmp, dcip);

    openclstepper.update(1);

  return 0;
}
