#ifndef LIBGEODECOMP_IO_QTWIDGETWRITER_H
#define LIBGEODECOMP_IO_QTWIDGETWRITER_H

#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_WITH_QT5

#include <libgeodecomp/io/plotter.h>
#include <libgeodecomp/io/writer.h>
#include <libgeodecomp/misc/clonable.h>
#include <libgeodecomp/misc/palette.h>
#include <libgeodecomp/misc/quickpalette.h>
#include <libgeodecomp/misc/sharedptr.h>

// Kill some warnings in system headers
#ifdef _MSC_BUILD
#pragma warning( push )
#pragma warning( disable : 4514 4710 4711 )
#endif

#ifdef __ICC
// disabling this warning as implicit type conversion is exactly our goal here:
#pragma warning push
#pragma warning (disable: 2304)
#endif

#ifdef __GNUC__
#ifdef __CUDACC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif
#endif

#include <QtGui/QBackingStore>
#include <QtGui/QPainter>
#include <QtGui/QResizeEvent>
#include <QtGui/QWindow>

#ifdef __GNUC__
#ifdef __CUDACC__
#pragma GCC diagnostic pop
#endif
#endif

#ifdef __ICC
#pragma warning pop
#endif

#ifdef _MSC_BUILD
#pragma warning( pop )
#endif

namespace LibGeoDecomp {

class QtWidgetWriterTest;

namespace QtWidgetWriterHelpers {

/**
 * Generic interface between Qt and LibGeoDecomp -- we need this class
 * as a QWindow can't be a template class.
 */
class Window : public QWindow
{
public:
    friend class LibGeoDecomp::QtWidgetWriterTest;

    Window(QWindow *parent = 0) :
        QWindow(parent),
        backingStore(new QBackingStore(this)),
        curImage(0, 0, QImage::Format_ARGB32),
        bufImage(0, 0, QImage::Format_ARGB32)
    {}

    ~Window()
    {
        delete backingStore;
    }

    void resizeImage(const Coord<2>& imageSize)
    {
        if (curImage.size() != QSize(imageSize.x(), imageSize.y())) {
            curImage = QImage(imageSize.x(), imageSize.y(), QImage::Format_ARGB32);
            bufImage = QImage(imageSize.x(), imageSize.y(), QImage::Format_ARGB32);
        }
    }

    void resizeEvent(QResizeEvent *event)
    {
        backingStore->resize(event->size());
    }

    bool event(QEvent *event)
    {
        // don't render 0-sized images
        if ((width() == 0) || (height() == 0)) {
            return true;
        }

        if ((event->type() == QEvent::Expose) ||
            (event->type() == QEvent::Paint) ||
            (event->type() == QEvent::Resize) ||
            (event->type() == QEvent::UpdateRequest)) {
            QPaintDevice *device = backingStore->paintDevice();
            QPainter painter(device);
            painter.drawImage(curImage.rect(), curImage);
        }

        return true;
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
        using std::swap;
        swap(curImage, bufImage);
    }

private:
    QBackingStore *backingStore;
    QImage curImage;
    QImage bufImage;
};

/**
 * Wraps a QPainter and exposes the painter interface expected by a Plotter.
 */
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
class QtWidgetWriter : public Clonable<Writer<CELL_TYPE>, QtWidgetWriter<CELL_TYPE, CELL_PLOTTER> >
{
public:
    friend class QtWidgetWriterTest;
    typedef typename Writer<CELL_TYPE>::GridType GridType;

    /**
     * This QtWriter will render a given member (e.g. &Cell::fooBar).
     * Colouring is handled by a predefined palette. The color range
     * is mapped to the value range defined by [minValue, maxValue].
     *
     * cellDimensions controls the size of the tiles in which a cell
     * will be rendered.
     */
    template<typename MEMBER>
    QtWidgetWriter(
        MEMBER CELL_TYPE:: *member,
        MEMBER minValue,
        MEMBER maxValue,
        const Coord<2>& cellDimensions = Coord<2>(8, 8),
        unsigned period = 1) :
        Clonable<Writer<CELL_TYPE>, QtWidgetWriter<CELL_TYPE, CELL_PLOTTER> >("", period),
        plotter(cellDimensions, CELL_PLOTTER(member, QuickPalette<MEMBER>(minValue, maxValue))),
        cellDimensions(cellDimensions),
        myWindow(new QtWidgetWriterHelpers::Window)
    {}

    /**
     * Creates a QtWriter which will render the values of the given
     * member variable. Color mapping is done with the help of the
     * custom palette object.
     */
    template<typename MEMBER, typename PALETTE>
    QtWidgetWriter(
        MEMBER CELL_TYPE:: *member,
        const PALETTE& palette,
        const Coord<2>& cellDimensions = Coord<2>(8, 8),
        unsigned period = 1) :
        Clonable<Writer<CELL_TYPE>, QtWidgetWriter<CELL_TYPE, CELL_PLOTTER> >("", period),
        plotter(cellDimensions, CELL_PLOTTER(member, palette)),
        cellDimensions(cellDimensions),
        myWindow(new QtWidgetWriterHelpers::Window)
    {}

    virtual void stepFinished(const GridType& grid, unsigned step, WriterEvent event)
    {
        Coord<2> gridDim(grid.dimensions());
        Coord<2> imageSize(
            gridDim.x() * cellDimensions.x(),
            gridDim.y() * cellDimensions.y());
        myWindow->resizeImage(imageSize);

        {
            QPainter qPainter(myWindow->getImage());
            QtWidgetWriterHelpers::PainterWrapper painter(&qPainter);
            CoordBox<2> viewport(Coord<2>(0, 0), myWindow->getImage()->size());
            plotter.plotGridInViewport(grid, painter, viewport);
        }
        myWindow->swapImages();
        myWindow->requestUpdate();
    }

    SharedPtr<QtWidgetWriterHelpers::Window>::Type window()
    {
        return myWindow;
    }

private:
    Plotter<CELL_TYPE, CELL_PLOTTER> plotter;
    Coord<2> cellDimensions;
    // we can't use multiple inheritance as Q_OBJECT doesn't support template classes.
    SharedPtr<QtWidgetWriterHelpers::Window>::Type myWindow;
};

}

#endif

#endif
