#ifndef _libgeodecomp_examples_flowingcanvas_canvasinitializer_h_
#define _libgeodecomp_examples_flowingcanvas_canvasinitializer_h_

#include <libgeodecomp/examples/flowingcanvas/canvascell.h>
#include <libgeodecomp/io/simpleinitializer.h>

namespace LibGeoDecomp {

class CanvasInitializer : public SimpleInitializer<CanvasCell>
{
public:
    CanvasInitializer() :
        // SimpleInitializer<CanvasCell>(Coord<2>(240, 135), 100)
        // SimpleInitializer<CanvasCell>(Coord<2>(320, 180), 100)
        SimpleInitializer<CanvasCell>(Coord<2>(384, 216), 100)
        // SimpleInitializer<CanvasCell>(Coord<2>(640, 360), 100)
    {}

    virtual void grid(GridBase<CanvasCell, 2> *ret)
    {
        CoordBox<2> box = ret->boundingBox();
        CoordBoxSequence<2> s = box.sequence();
        while (s.hasNext()) {
            Coord<2> c = s.next();
            bool setForce = false;
            FloatCoord<2> force;

            if ((c.x() == 100) && (c.y() >= 100) && (c.y() <= 200)) {
                setForce = true;
                force.c[1] = -1;
            }
            if ((c.x() == 200) && (c.y() >= 100) && (c.y() <= 200)) {
                setForce = true;
                force.c[1] = 1;
            }
            if ((c.y() == 100) && (c.x() >= 100) && (c.x() <= 200)) {
                setForce = true;
                force.c[0] = 1;
            }
            if ((c.y() == 200) && (c.x() >= 100) && (c.x() <= 200)) {
                setForce = true;
                force.c[0] = -1;
            }

            ret->at(c) = CanvasCell(c, setForce, force);
        }
    }
};

}

#endif
