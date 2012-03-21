#ifndef _libgeodecomp_examples_flowingcanvas_canvasinitializer_h_
#define _libgeodecomp_examples_flowingcanvas_canvasinitializer_h_

#include <libgeodecomp/examples/flowingcanvas/canvascell.h>
#include <libgeodecomp/io/simpleinitializer.h>

namespace LibGeoDecomp {

class CanvasInitializer : public SimpleInitializer<CanvasCell>
{
public:
    CanvasInitializer() :
        SimpleInitializer<CanvasCell>(Coord<2>(320, 180), 100)
    {}

    virtual void grid(GridBase<CanvasCell, 2> *ret)
    {
        CoordBox<2> box = ret->boundingBox();
        CoordBoxSequence<2> s = box.sequence();
        while (s.hasNext()) {
            Coord<2> c = s.next();
            bool setForce = false;
            FloatCoord<2> force;

            if ((c.x() == 140) && (c.y() >= 80) && (c.y() <= 160)) {
                setForce = true;
                force.c[1] = -1;
            }
            if ((c.x() == 220) && (c.y() >= 80) && (c.y() <= 160)) {
                setForce = true;
                force.c[1] = 1;
            }
            if ((c.y() == 80) && (c.x() >= 140) && (c.x() <= 220)) {
                setForce = true;
                force.c[0] = 1;
            }
            if ((c.y() == 160) && (c.x() >= 140) && (c.x() <= 220)) {
                setForce = true;
                force.c[0] = -1;
            }

            ret->at(c) = CanvasCell(c, setForce, force, rand() % CanvasCell::MAX_SPAWN_COUNTDOWN);
        }
    }
};

}

#endif
