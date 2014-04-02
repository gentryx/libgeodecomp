#include <libgeodecomp/config.h>
#include <libgeodecomp/io/selector.h>
#include <libgeodecomp/io/silowriter.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/misc/stdcontaineroverloads.h>
#include <libgeodecomp/misc/tempfile.h>

#ifdef LIBGEODECOMP_WITH_SILO
#ifdef LIBGEODECOMP_WITH_QT

#include <Python.h>
#include <QtGui/QApplication>
#include <QtGui/QColor>
#include <QtGui/QImage>

#endif
#endif

using namespace LibGeoDecomp;

class DummyParticle
{
public:
    DummyParticle(const FloatCoord<2> pos) :
        pos(pos)
    {
        coords << pos + FloatCoord<2>(-5, -5)
               << pos + FloatCoord<2>( 5, -5)
               << pos + FloatCoord<2>( 5,  5)
               << pos + FloatCoord<2>(-5,  5);
    }

    FloatCoord<2> getPoint() const
    {
        return pos;
    }

    std::vector<FloatCoord<2> > getShape() const
    {
        return coords;
    }

    FloatCoord<2> pos;
    std::vector<FloatCoord<2> > coords;
};

typedef std::vector<DummyParticle> ParticleVec;

class CellWithPointMesh
{
public:
    typedef DummyParticle Cargo;

    class API :
        public APITraits::HasCustomRegularGrid,
        public APITraits::HasPointMesh,
        public APITraits::HasUnstructuredGrid
    {
    public:
        static FloatCoord<2> getRegularGridSpacing()
        {
            return FloatCoord<2>(20, 10);
        };
    };

    CellWithPointMesh(double dummyValue = 0) :
        dummyValue(dummyValue)
    {}

    ParticleVec::const_iterator begin() const
    {
        return particles.begin();
    }

    ParticleVec::const_iterator end() const
    {
        return particles.end();
    }

    std::size_t size() const
    {
        return particles.size();
    }

    ParticleVec particles;
    double dummyValue;
};

class ParticleFilterBase : public Selector<DummyParticle>::Filter<FloatCoord<2>, double>
{
public:
    void copyStreakInImpl(const double *first, const double *last, FloatCoord<2> *target)
    {
        // left blank as it's not needed in our tests
    }

    void copyStreakOutImpl(const FloatCoord<2> *first, const FloatCoord<2> *last, double *target)
    {
        // left blank as it's not needed in our tests
    }

    void copyMemberInImpl(
        const double *source, DummyParticle *target, int num, FloatCoord<2> DummyParticle:: *memberPointer)
    {
        // left blank as it's not needed in our tests
    }
};

class ParticleFilterX : public ParticleFilterBase
{
    void copyMemberOutImpl(
        const DummyParticle *source, double *target, int num, FloatCoord<2> DummyParticle:: *memberPointer)
    {
        for (int i = 0; i < num; ++i) {
            target[i] = (source[i].*memberPointer)[0];
        }
    }
};

class ParticleFilterY : public ParticleFilterBase
{
    void copyMemberOutImpl(
        const DummyParticle *source, double *target, int num, FloatCoord<2> DummyParticle:: *memberPointer)
    {
        for (int i = 0; i < num; ++i) {
            target[i] = (source[i].*memberPointer)[1];
        }
    }
};

namespace LibGeoDecomp {

class SiloWriterTest : public CxxTest::TestSuite
{
public:
    void setUp()
    {
#ifdef LIBGEODECOMP_WITH_QT
        int argc = 0;
        char **argv = 0;
        app.reset(new QApplication(argc, argv));
#endif

        prefix = TempFile::serial("silowriter_test") + "foo";
        siloFile = prefix + ".00123.silo";

        remove((prefix + "0000.png").c_str());
        remove((prefix + "0001.png").c_str());
        remove((prefix + "0002.png").c_str());
        remove((prefix + "0003.png").c_str());
        remove((prefix + "0004.png").c_str());
    }

    void tearDown()
    {
        remove((prefix + "0000.png").c_str());
        remove((prefix + "0001.png").c_str());
        remove((prefix + "0002.png").c_str());
        remove((prefix + "0003.png").c_str());
        remove((prefix + "0004.png").c_str());
        remove(siloFile.c_str());
#ifdef LIBGEODECOMP_WITH_QT
        app.reset();
#endif
    }

#ifdef LIBGEODECOMP_WITH_SILO
#ifdef LIBGEODECOMP_WITH_VISIT
#ifdef LIBGEODECOMP_WITH_QT

    typedef std::map<QRgb, int> Histogram;

    void render(std::string visitScript)
    {
        // init grid
        Coord<2> dim(10, 5);
        CoordBox<2> box(Coord<2>(), dim);
        FloatCoord<2> quadrantDim = APITraits::SelectRegularGrid<CellWithPointMesh>::value();

        Grid<CellWithPointMesh> grid(dim);
        int counter = 0;

        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            grid[*i] = CellWithPointMesh(counter++);
            FloatCoord<2> center =
                FloatCoord<2>(*i).scale(quadrantDim) +
                quadrantDim * 0.5;
            grid[*i].particles << DummyParticle(center);
        }

        // dump silo file
        SiloWriter<CellWithPointMesh> writer(prefix, 1);
        writer.addSelector(Selector<CellWithPointMesh>(&CellWithPointMesh::dummyValue, "dummyValue"));

        boost::shared_ptr<Selector<DummyParticle>::FilterBase> filterX(new ParticleFilterX());
        boost::shared_ptr<Selector<DummyParticle>::FilterBase> filterY(new ParticleFilterY());

        writer.addSelectorForPointMesh(
            Selector<DummyParticle>(&DummyParticle::pos, "posX", filterX));
        writer.addSelectorForUnstructuredGrid(
            Selector<DummyParticle>(&DummyParticle::pos, "posY", filterY));
        writer.stepFinished(grid, 123, WRITER_INITIALIZED);

        // plot
        Py_Initialize();
        PyRun_SimpleString(visitScript.c_str());
        Py_Finalize();

        remove(siloFile.c_str());
    }

    Histogram loadImage(std::string suffix)
    {
        Histogram ret;

        std::string imageFile = prefix + suffix;

        QImage result;
        result.load(QString(imageFile.c_str()));
        TS_ASSERT_EQUALS(QSize(1024, 1024), result.size());

        for (int y = 0; y < 1024; ++y) {
            for (int x = 0; x < 1024; ++x) {
                ret[result.pixel(x, y)] += 1;
            }
        }

        remove(imageFile.c_str());

        return ret;
    }

#endif
#endif
#endif

    void testBasic()
    {
#ifdef LIBGEODECOMP_WITH_SILO
#ifdef LIBGEODECOMP_WITH_VISIT
#ifdef LIBGEODECOMP_WITH_QT

        QColor white(255, 255, 255);
        QColor black(0, 0, 0);
        QColor red(  255, 0, 0);
        QColor green(0, 255, 0);
        QColor blue( 0, 0, 255);

        std::stringstream buf;
        buf << "import re\n"
            << "import os\n"
            << "import visit\n"
            << "\n"
            << "simfile = \"" << siloFile << "\"\n"
            << "\n"
            << "visit.LaunchNowin ()\n"
            << "visit.OpenDatabase(simfile)\n"
            << "attributes = visit.SaveWindowAttributes()\n"
            << "attributes.fileName = \"" << prefix << "\"\n"
            << "attributes.format = attributes.PNG\n"
            << "visit.SetSaveWindowAttributes(attributes)\n";
        // first image
        buf << "visit.AddPlot(\"Mesh\", \"regular_grid\")\n"
            << "visit.DrawPlots()\n"
            << "visit.SaveWindow()\n"
            << "visit.DeleteAllPlots()\n";
        // second image
        buf << "visit.AddPlot(\"Mesh\", \"point_mesh\")\n"
            << "visit.AddPlot(\"Mesh\", \"regular_grid\")\n"
            << "visit.DrawPlots()\n"
            << "visit.SaveWindow()\n"
            << "visit.DeleteAllPlots()\n";
        // third image
        buf << "visit.AddPlot(\"Mesh\", \"regular_grid\")\n"
            << "visit.AddPlot(\"Mesh\", \"point_mesh\")\n"
            << "visit.AddPlot(\"Pseudocolor\", \"dummyValue\")\n"
            << "visit.DrawPlots()\n"
            << "visit.SaveWindow()\n"
            << "visit.DeleteAllPlots()\n";
        // fourth image
        buf << "visit.AddPlot(\"Mesh\", \"regular_grid\")\n"
            << "visit.AddPlot(\"Pseudocolor\", \"posX\")\n"
            << "visit.DrawPlots()\n"
            << "visit.SaveWindow()\n"
            << "visit.DeleteAllPlots()\n";
        // fifth image
        buf << "visit.AddPlot(\"Mesh\", \"regular_grid\")\n"
            << "visit.AddPlot(\"Mesh\", \"point_mesh\")\n"
            << "visit.AddPlot(\"Pseudocolor\", \"posY\")\n"
            << "visit.DrawPlots()\n"
            << "visit.SaveWindow()\n"
            << "visit.DeleteAllPlots()\n";
        render(buf.str());

        Histogram histogram1 = loadImage("0000.png");

        TS_ASSERT(histogram1[white.rgb()] > 900000);

        Histogram histogram2 = loadImage("0001.png");

        TS_ASSERT(histogram1[white.rgb()] > histogram2[white.rgb()]);
        // point mesh should add 50 dots a 2x2 pixels plus a label
        TS_ASSERT((histogram2[black.rgb()] - histogram1[black.rgb()]) > 200);

        Histogram histogram3 = loadImage("0002.png");

        TS_ASSERT(histogram3[white.rgb()] > 800000);
        TS_ASSERT_EQUALS(histogram3[red.rgb()  ], 2700);


        Histogram histogram4 = loadImage("0003.png");

        TS_ASSERT(histogram1[white.rgb()] > histogram4[white.rgb()]);
        TS_ASSERT(histogram1[black.rgb()] < histogram4[black.rgb()]);
        TS_ASSERT(histogram4[red.rgb()  ] > 20);

        Histogram histogram5 = loadImage("0004.png");

        TS_ASSERT(histogram4[red.rgb()  ] > 20);
        TS_ASSERT(histogram4[white.rgb()] < histogram1[white.rgb()]);

#endif
#endif
#endif
    }

private:
#ifdef LIBGEODECOMP_WITH_QT
    boost::shared_ptr<QApplication> app;
#endif
    std::string prefix;
    std::string siloFile;
};

}
