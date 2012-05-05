#ifndef _libgeodecomp_examples_flowingcanvas_canvasinitializer_h_
#define _libgeodecomp_examples_flowingcanvas_canvasinitializer_h_

#include <QPixmap>
#include <libgeodecomp/examples/flowingcanvas/canvascell.h>
#include <libgeodecomp/io/simpleinitializer.h>

namespace LibGeoDecomp {

class CanvasInitializer : public SimpleInitializer<CanvasCell>
{
public:
    CanvasInitializer() :
        SimpleInitializer<CanvasCell>(Coord<2>(320, 180), 100)
        // SimpleInitializer<CanvasCell>(Coord<2>(640, 360), 100)
    {}

    virtual void grid(GridBase<CanvasCell, 2> *ret)
    {
        CoordBox<2> box = ret->boundingBox();

        // fixme:
        // QImage source = QPixmap("starry_night.png").scaled(box.dimensions.x(), box.dimensions.y()).toImage();
        // source.convertToFormat(QImage::Format_ARGB32);

        CoordBoxSequence<2> s = box.sequence();
        while (s.hasNext()) {
            Coord<2> c = s.next();
            bool setForce = false;
            FloatCoord<2> force;

            if ((c.x() == 140) && (c.y() >= 80) && (c.y() <= 160)) {
                setForce = true;
                force[1] = -1;
            }
            if ((c.x() == 220) && (c.y() >= 80) && (c.y() <= 160)) {
                setForce = true;
                force[1] = 1;
            }
            if ((c.y() == 80) && (c.x() >= 140) && (c.x() <= 220)) {
                setForce = true;
                force[0] = 1;
            }
            if ((c.y() == 160) && (c.x() >= 140) && (c.x() <= 220)) {
                setForce = true;
                force[0] = -1;
            }

            // fixme
            // unsigned pixel = source.pixel(c.x(), c.y());
            unsigned pixel = (0xff << 24) + ((c.x() * 255 / 400) << 16) + ((c.y() * 255 / 200) << 8) + 0;
            ret->at(c) = CanvasCell(pixel, c, setForce, force, rand() % CanvasCell::MAX_SPAWN_COUNTDOWN);
        }
    }
};

}

#endif
