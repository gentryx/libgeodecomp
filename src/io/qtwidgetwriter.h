#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_QT

#ifndef LIBGEODECOMP_IO_QTWIDGETWRITER_H
#define LIBGEODECOMP_IO_QTWIDGETWRITER_H

#include <QtGui/QPainter>
#include <QtGui/QWidget>

#include <libgeodecomp/io/plotter.h>
#include <libgeodecomp/io/writer.h>

namespace LibGeoDecomp {

namespace QtWidgetWriterHelpers {

class Widget : public QWidget
{
    Q_OBJECT
public:
    Widget() :
        image(1000, 1000, QImage::Format_ARGB32),
	counter(0)
    {}

    void paintEvent(QPaintEvent * /* event */)
    {
        QPainter painter(this);

        painter.drawImage(0, 0, image);

        painter.setPen(Qt::green);
        painter.drawText(32, 32, "Frame " + QString::number(counter));
        ++counter;
    }

    QImage *getImage()
    {
        return &image;
    }

private:
    QImage image;
    int counter;
};

class PainterWrapper
{
public:
    PainterWrapper(QPainter *painter) :
        painter(painter)
    {
        painter->save();
    }

    void moveTo(const Coord<2>& coord)
    {
        painter->restore();
        painter->save();
        painter->translate(coord.x(), coord.y());
    }

    void fillRect(int x, int y, int dimX, int dimY, Color color)
    {
        painter->fillRect(400, 400, 100, 100, 0x00ffff00);
        painter->fillRect(x, y, dimX / 2, dimY / 2, color.rgb);
    }

  private:
    QPainter *painter;
};

}
template<typename CELL_TYPE, typename CELL_PLOTTER>
class QtWidgetWriter : public Writer<CELL_TYPE>
{
public:
    typedef typename Writer<CELL_TYPE>::GridType GridType;

    QtWidgetWriter(const Coord<2>& cellDimensions = Coord<2>(8, 8)) :
	cellDimensions(cellDimensions)
    {}

    virtual void stepFinished(const GridType& grid, unsigned step, WriterEvent event)
    {
        QPainter qPainter(myWidget.getImage());
        QtWidgetWriterHelpers::PainterWrapper painter(&qPainter);
        Plotter<CELL_TYPE, CELL_PLOTTER> plotter(cellDimensions);
        plotter.plotGrid(grid, painter);
        myWidget.update();
    }

    QWidget *widget()
    {
	return &myWidget;
    }

private:
    Coord<2> cellDimensions;
    // we can't use multiple inheritance as Q_OBJECT doesn't support template classes.
    QtWidgetWriterHelpers::Widget myWidget;
};

}

#endif

#endif
