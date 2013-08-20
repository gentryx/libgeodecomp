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

class DummyCell : public OpenCLCellInterface<DummyCell, MyCell> {
  public:
    static const int DIMENSIONS = 3;
    typedef Stencils::VonNeumann<3, 1> Stencil;
    class API : public CellAPITraits::Base {};
    typedef Topologies::Cube<3>::Topology Topology;

    DummyCell(void) {}

    DummyCell(int x, int y, int z)
    {
      myCellData.x = x;
      myCellData.y = y;
      myCellData.z = z;
    }

    static inline unsigned nanoSteps() { return 1; }
    template<typename COORD_MAP>
      void update(const COORD_MAP& neighborhood, const unsigned& nanoStep) {}


    static std::string kernel_file() { return "./test.cl"; }
    static std::string kernel_function() { return "dummy_test"; }
    static std::string cl_struct_code() { return STRINGIFY(MYCELL_STRUCT); }
    static size_t sizeof_data() { return sizeof(MyCell); }
    MyCell * data() { return &myCellData; }

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
            ret->set(c, DummyCell(x, y, z));
          }
        }
      }
    }
};

int main(int argc, char **argv)
{

  boost::shared_ptr<HiParSimulator::PartitionManager<DummyCell::Topology>>
    pmp(new HiParSimulator::PartitionManager<DummyCell::Topology>(
          CoordBox<3>(Coord<3>(0,0,0), Coord<3>(2,2,2))));

  boost::shared_ptr<DummyCellInitializer> dcip(new DummyCellInitializer);

  HiParSimulator::OpenCLStepper<DummyCell, MyCell> openclstepper(0, 0, pmp, dcip);

  openclstepper.update(1);

  return 0;
}
