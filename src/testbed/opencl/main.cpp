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
    static std::string kernel_function() { return "add_test"; }
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

class JacobiCell : public OpenCLCellInterface<JacobiCell, double> {
  public:
    static const int DIMENSIONS = 3;
    typedef Stencils::VonNeumann<3, 1> Stencil;
    class API : public CellAPITraits::Base {};
    typedef Topologies::Cube<3>::Topology Topology;

    JacobiCell(void) {}
    JacobiCell(double temp) : m_temp(temp) {}

    static inline unsigned nanoSteps() { return 1; }

    template<typename COORD_MAP>
      void update(const COORD_MAP& neighborhood, const unsigned& nanoStep) {}

    static std::string kernel_file() { return "./jacobi3d.cl"; }
    static std::string kernel_function() { return "update"; }
    double * data() { return &m_temp; }

    double m_temp;
};

class JacobiCellInitializer : public SimpleInitializer<JacobiCell> {
  public:
    using SimpleInitializer<JacobiCell>::gridDimensions;

    JacobiCellInitializer(void) : SimpleInitializer<JacobiCell>(Coord<3>(2, 2, 2))
    {}

    virtual void grid(GridBase<JacobiCell, 3> *ret)
    {
      Coord<3> offset = Coord<3>(0, 0, 0);

      for (int z = 0; z < gridDimensions().z(); ++z) {
        for (int y = 0; y < gridDimensions().y(); ++y) {
          for (int x = 0; x < gridDimensions().x(); ++x) {
            Coord<3> c = offset + Coord<3>(x, y, z);
            ret->set(c, JacobiCell(0.99999999999));
          }
        }
      }
    }
};

int main(int argc, char **argv)
{
  // boost::shared_ptr<HiParSimulator::PartitionManager<DummyCell::Topology>>
  //   pmp(new HiParSimulator::PartitionManager<DummyCell::Topology>(
  //         CoordBox<3>(Coord<3>(0,0,0), Coord<3>(2,2,2))));

  // boost::shared_ptr<DummyCellInitializer> dcip(new DummyCellInitializer);

  // HiParSimulator::OpenCLStepper<DummyCell, MyCell> openclstepper(0, 0, pmp, dcip);

  // openclstepper.update(2);

  // auto & grid = openclstepper.grid();

  // std::cerr << "result" << std::endl;
  // for (auto & p : dcip->gridBox()) {
  //   std::cerr << "(" << grid.get(p).myCellData.x << ", "
  //                    << grid.get(p).myCellData.y << ", "
  //                    << grid.get(p).myCellData.z << ")"
  //                    << " @ " << p
  //                    << std::endl;
  // }

  boost::shared_ptr<HiParSimulator::PartitionManager<JacobiCell::Topology>>
    pmp(new HiParSimulator::PartitionManager<JacobiCell::Topology>(
          CoordBox<3>(Coord<3>(0,0,0), Coord<3>(2,2,2))));

  boost::shared_ptr<JacobiCellInitializer> dcip(new JacobiCellInitializer);

  HiParSimulator::OpenCLStepper<JacobiCell, double> openclstepper(0, 0, pmp, dcip);

  openclstepper.update(2);



  return 0;
}
