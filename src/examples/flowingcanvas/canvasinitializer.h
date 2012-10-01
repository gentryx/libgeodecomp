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

    virtual void grid(GridBase<CanvasCell, 2> *ret) const
    {
        CoordBox<2> box = ret->boundingBox();

        // fixme:
        // QImage source = QPixmap("starry_night.png").scaled(box.dimensions.x(), box.dimensions.y()).toImage();
        // source.convertToFormat(QImage::Format_ARGB32);

        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            bool setForce = false;
            FloatCoord<2> force;

            if ((i->x() == 140) && (i->y() >= 80) && (i->y() <= 160)) {
                setForce = true;
                force[1] = -1;
            }
            if ((i->x() == 220) && (i->y() >= 80) && (i->y() <= 160)) {
                setForce = true;
                force[1] = 1;
            }
            if ((i->y() == 80) && (i->x() >= 140) && (i->x() <= 220)) {
                setForce = true;
                force[0] = 1;
            }
            if ((i->y() == 160) && (i->x() >= 140) && (i->x() <= 220)) {
                setForce = true;
                force[0] = -1;
            }

            // fixme
            // unsigned pixel = source.pixel(i->x(), i->y());
            unsigned pixel = (0xff << 24) + 
                ((i->x() * 255 / 400) << 16) + 
                ((i->y() * 255 / 200) << 8) + 0;
            ret->at(*i) = CanvasCell(pixel, *i, setForce, force, 
                                     rand() % CanvasCell::MAX_SPAWN_COUNTDOWN);
        }
    }
};

}

#endif
