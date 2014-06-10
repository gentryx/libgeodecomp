#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_QT

#ifndef LIBGEODECOMP_IO_QTWIDGETWRITER_H
#define LIBGEODECOMP_IO_QTWIDGETWRITER_H

#include <libgeodecomp/io/plotter.h>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/misc/palette.h>
#include <libgeodecomp/misc/quickpalette.h>

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif

#include <QtGui/QResizeEvent>
#include <QtGui/QPainter>
#include <QtGui/QWidget>

#ifdef __ICC
#pragma warning pop
#endif

namespace LibGeoDecomp {

class QtWidgetWriterTest;

namespace QtWidgetWriterHelpers {

class Widget : public QWidget
{
public:
    friend class LibGeoDecomp::QtWidgetWriterTest;

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
    explicit PainterWrapper(QPainter *painter) :
        painter(painter)
    {}

    ~PainterWrapper()
    {
        painter->translate(-translation.x(), -translation.y());
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

/**
 * This Writer displays 2D data via a Qt GUI element. A plotter can be
 * used to customize the rendering.
 */
template<typename CELL_TYPE, typename CELL_PLOTTER = SimpleCellPlotter<CELL_TYPE> >
class QtWidgetWriter : public Writer<CELL_TYPE>
{
public:
    friend class QtWidgetWriterTest;
    typedef typename Writer<CELL_TYPE>::GridType GridType;

    template<typename MEMBER>
    QtWidgetWriter(
        MEMBER CELL_TYPE:: *member,
        const Palette<MEMBER>& palette,
        const Coord<2>& cellDimensions = Coord<2>(8, 8),
        unsigned period = 1) :
        Writer<CELL_TYPE>("", period),
        plotter(cellDimensions, CELL_PLOTTER(member, palette)),
        cellDimensions(cellDimensions)
    {}

    template<typename MEMBER>
    QtWidgetWriter(
        MEMBER CELL_TYPE:: *member,
        MEMBER minValue,
        MEMBER maxValue,
        const Coord<2>& cellDimensions = Coord<2>(8, 8),
        unsigned period = 1) :
        Writer<CELL_TYPE>("", period),
        plotter(cellDimensions, CELL_PLOTTER(member, QuickPalette<MEMBER>(minValue, maxValue))),
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
    Plotter<CELL_TYPE, CELL_PLOTTER> plotter;
    Coord<2> cellDimensions;
    // we can't use multiple inheritance as Q_OBJECT doesn't support template classes.
    QtWidgetWriterHelpers::Widget myWidget;
};

}

#endif

#endif
