#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_QT

#ifndef LIBGEODECOMP_IO_QTWIDGETWRITER_H
#define LIBGEODECOMP_IO_QTWIDGETWRITER_H

#include <QtGui/QResizeEvent>
#include <QtGui/QPainter>
#include <QtGui/QWidget>

#include <libgeodecomp/io/plotter.h>
#include <libgeodecomp/io/writer.h>

namespace LibGeoDecomp {

namespace QtWidgetWriterHelpers {

class Widget : public QWidget
{
public:
    Widget() :
        curImage(0, 0, QImage::Format_ARGB32),
        bufImage(0, 0, QImage::Format_ARGB32)
    {}

    void resizeImage(const Coord<2>& imageSize)
    {
        if (curImage.size() != QSize(imageSize.x(), imageSize.y())) {
            curImage = QImage(imageSize.x(), imageSize.y(), QImage::Format_ARGB32);
            bufImage = QImage(imageSize.x(), imageSize.y(), QImage::Format_ARGB32);
        }
    }

    void paintEvent(QPaintEvent *event)
    {
        QPainter painter(this);
        painter.drawImage(event->rect(), curImage);
    }

    Coord<2> dimensions() const
    {
        return Coord<2>(width(), height());
    }

    QImage *getImage()
    {
        return &bufImage;
    }

    void swapImages()
    {
        std::swap(curImage, bufImage);
    }

private:
    QImage curImage;
    QImage bufImage;
};

class PainterWrapper
{
public:
    PainterWrapper(QPainter *painter) :
        painter(painter)
    {}

    ~PainterWrapper()
    {
        // painter->translate(-translation.x(), -translation.y());
    }

    void moveTo(const Coord<2>& coord)
    {
        painter->translate(coord.x() - translation.x(),
                           coord.y() - translation.y());
        translation = coord;
    }

    void fillRect(int x, int y, int dimX, int dimY, Color color)
    {
        painter->fillRect(x, y, dimX, dimY, color.rgb);
    }

private:
    QPainter *painter;
    Coord<2> translation;
};

}
template<typename CELL_TYPE, typename CELL_PLOTTER>
class QtWidgetWriter : public Writer<CELL_TYPE>
{
public:
    typedef typename Writer<CELL_TYPE>::GridType GridType;

    QtWidgetWriter(const Coord<2>& cellDimensions = Coord<2>(8, 8)) :
        Writer<CELL_TYPE>("", 1),
        cellDimensions(cellDimensions)
    {}

    virtual void stepFinished(const GridType& grid, unsigned step, WriterEvent event)
    {
        Coord<2> gridDim(grid.dimensions());
        Coord<2> imageSize(
            gridDim.x() * cellDimensions.x(),
            gridDim.y() * cellDimensions.y());
        myWidget.resizeImage(imageSize);

        QPainter qPainter(myWidget.getImage());

        {
            QtWidgetWriterHelpers::PainterWrapper painter(&qPainter);
            Plotter<CELL_TYPE, CELL_PLOTTER> plotter(cellDimensions);
            CoordBox<2> viewport(Coord<2>(0, 0), myWidget.getImage()->size());
            plotter.plotGridInViewport(grid, painter, viewport);
        }

        myWidget.swapImages();
        myWidget.update();
    }

    QtWidgetWriterHelpers::Widget *widget()
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
