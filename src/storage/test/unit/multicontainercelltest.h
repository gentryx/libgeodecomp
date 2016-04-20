#include <libgeodecomp/storage/grid.h>
#include <libgeodecomp/storage/boxcell.h>
#include <libgeodecomp/storage/multicontainercell.h>
#include <libgeodecomp/storage/updatefunctor.h>

#include <cxxtest/TestSuite.h>

using namespace LibGeoDecomp;

namespace LibGeoDecomp {

DECLARE_MULTI_CONTAINER_CELL(
    DummyContainer,
    DummyContainer,
    (((ContainerCell<std::string, 5>))(labels))
    (((ContainerCell<double,      7>))(prices)) )

class APIProvider
{
public:
    class API : public LibGeoDecomp::APITraits::HasTorusTopology<3>,
                public LibGeoDecomp::APITraits::HasPointMesh,
                public LibGeoDecomp::APITraits::HasStencil<LibGeoDecomp::Stencils::Moore<3, 1> >
    {};
};

class SpawningParticleBlue;

/**
 * Another simple test particle which simply spawns new particles
 */
class SpawningParticleRed
{
public:
    class API : public APIProvider::API
    {};

    explicit SpawningParticleRed(
        const FloatCoord<3>& pos = FloatCoord<3>(),
        const int numRedParticlesToBeSpawned = 0,
        const int numBlueParticlesToBeSpawned = 0) :
        pos(pos),
        numRedParticlesToBeSpawned(numRedParticlesToBeSpawned),
        numBlueParticlesToBeSpawned(numBlueParticlesToBeSpawned)
    {}

    inline const FloatCoord<3>& getPos() const
    {
        return pos;
    }

    template<typename HOOD>
    inline void update(HOOD& hood, const int nanoStep)
    {
        for (int i = 0; i < numRedParticlesToBeSpawned; ++i) {
            hood->red << SpawningParticleRed(pos, 0);
        }
        for (int i = 0; i < numBlueParticlesToBeSpawned; ++i) {
            hood->blue << SpawningParticleBlue(pos, 0);
        }
    }

private:
    FloatCoord<3> pos;
    int numRedParticlesToBeSpawned;
    int numBlueParticlesToBeSpawned;
};

/**
 * Another simple test particle which simply spawns new particles
 */
class SpawningParticleBlue
{
public:
    class API : public APIProvider::API
    {};

    explicit SpawningParticleBlue(
        const FloatCoord<3>& pos = FloatCoord<3>(),
        const int numRedParticlesToBeSpawned = 0,
        const int numBlueParticlesToBeSpawned = 0) :
        pos(pos),
        numRedParticlesToBeSpawned(numRedParticlesToBeSpawned),
        numBlueParticlesToBeSpawned(numBlueParticlesToBeSpawned)
    {}

    inline const FloatCoord<3>& getPos() const
    {
        return pos;
    }

    template<typename HOOD>
    inline void update(HOOD& hood, const int nanoStep)
    {
        for (int i = 0; i < numRedParticlesToBeSpawned; ++i) {
            hood->red << SpawningParticleRed(pos, 0);
        }
        for (int i = 0; i < numBlueParticlesToBeSpawned; ++i) {
            hood->blue << SpawningParticleBlue(pos, 0);
        }
    }

private:
    FloatCoord<3> pos;
    int numRedParticlesToBeSpawned;
    int numBlueParticlesToBeSpawned;
};

DECLARE_MULTI_CONTAINER_CELL(
    CellWithSpawningParticles,
    APIProvider,
    (((BoxCell<FixedArray<SpawningParticleRed,   16> >))(red))
    (((BoxCell<FixedArray<SpawningParticleBlue, 221> >))(blue)) )


typedef std::vector<std::pair<std::string, std::string> > LogType;
LogType multiContainerCellTestLog;

class SimpleElement;

class SimpleNode
{
public:
    explicit SimpleNode(const std::string& cargo = "") :
        cargo(cargo)
    {}

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, int nanoStep);

    std::string cargo;
};

class SimpleElement
{
public:
    explicit SimpleElement(const std::string& cargo = "") :
        cargo(cargo)
    {}

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, int nanoStep)
    {
        for (int i = 0; i < 20; ++i) {
            try {
                const SimpleNode& node = hood.nodes[i];
                multiContainerCellTestLog << std::make_pair(cargo, node.cargo);
            } catch(const std::logic_error& exception) {
                // intentionally left blank
            }
        }

        for (int i = 0; i < 20; ++i) {
            try {
                const SimpleElement& element = hood.elements[i];
                multiContainerCellTestLog << std::make_pair(cargo, element.cargo);
            } catch(const std::logic_error& exception) {
                // intentionally left blank
            }
        }
    }

    std::string cargo;
};

template<typename NEIGHBORHOOD>
void SimpleNode::update(const NEIGHBORHOOD& hood, int nanoStep)
{
    for (int i = 0; i < 20; ++i) {
        try {
            const SimpleNode& node = hood.nodes[i];
            multiContainerCellTestLog << std::make_pair(cargo, node.cargo);
        } catch(const std::logic_error& exception) {
            // intentionally left blank
        }
    }

    for (int i = 0; i < 20; ++i) {
        try {
            const SimpleElement& element = hood.elements[i];
            multiContainerCellTestLog << std::make_pair(cargo, element.cargo);
        } catch(const std::logic_error& exception) {
            // intentionally left blank
        }
    }
}

class SimpleParticle
{
public:
    explicit SimpleParticle(const double x = 0, const double y = 0) :
        pos(x, y)
    {}

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, int nanoStep)
    {
        seenNeighbors = 0;
        seenElements = 0;

        for (typename NEIGHBORHOOD::AdapterHelper3::Iterator i = hood.particles.begin();
             i != hood.particles.end();
             ++i) {
            ++seenNeighbors;
        }

        try {
            hood.elements[1024];
            ++seenElements;
        }  catch(const std::logic_error& exception) {
            // intentionally left blank
        }
    }

    const FloatCoord<2>& getPos() const
    {
        return pos;
    }

    int seenNeighbors;
    int seenElements;

private:
    FloatCoord<2> pos;
};


DECLARE_MULTI_CONTAINER_CELL(
    SimpleContainer,
    SimpleContainer,
    (((ContainerCell<SimpleNode,    30>))(nodes))
    (((ContainerCell<SimpleElement, 10>))(elements)) )

DECLARE_MULTI_CONTAINER_CELL(
    AnotherSimpleContainer,
    AnotherSimpleContainer,
    (((ContainerCell<SimpleNode,    30>))(nodes))
    (((BoxCell<FixedArray<SimpleParticle, 20> >))(particles))
    (((ContainerCell<SimpleElement, 10>))(elements)) )


class MultiContainerCellTest : public CxxTest::TestSuite
{
public:
    void testConstructionAndAccess()
    {
        DummyContainer cell;
        cell.labels.insert(10, "foo");
        cell.labels.insert(11, "bar");
        cell.labels.insert(15, "goo");

        cell.prices.insert(10, -666);
        cell.prices.insert(11, -0.11);
        cell.prices.insert(12, -0.12);
        cell.prices.insert(13, -0.13);
        cell.prices.insert(10, -0.10);

        TS_ASSERT_EQUALS(cell.labels.size(), std::size_t(3));
        TS_ASSERT_EQUALS(*cell.labels[10], "foo");
        TS_ASSERT_EQUALS(*cell.labels[11], "bar");
        TS_ASSERT_EQUALS(*cell.labels[15], "goo");

        TS_ASSERT_EQUALS(cell.prices.size(), std::size_t(4));
        TS_ASSERT_EQUALS(*cell.prices[10], -0.10);
        TS_ASSERT_EQUALS(*cell.prices[11], -0.11);
        TS_ASSERT_EQUALS(*cell.prices[12], -0.12);
        TS_ASSERT_EQUALS(*cell.prices[13], -0.13);
    }

    void testUpdate()
    {
        Coord<2> dim(10, 5);
        Grid<SimpleContainer> gridOld(dim);
        Grid<SimpleContainer> gridNew(dim);

        SimpleContainer c;
        c.nodes.insert(1, SimpleNode("Node1"));
        c.nodes.insert(5, SimpleNode("Node5a"));
        gridOld[Coord<2>(3, 3)] = c;

        SimpleContainer d;
        d.nodes.insert(6, SimpleNode("Node6"));
        d.elements.insert(1, SimpleElement("Element1"));
        d.elements.insert(7, SimpleElement("Element7"));
        d.elements.insert(9, SimpleElement("Element9"));
        gridOld[Coord<2>(3, 4)] = d;

        SimpleContainer e;
        e.nodes.insert(10, SimpleNode("Node10"));
        e.nodes.insert(11, SimpleNode("Node11"));
        e.elements.insert(5, SimpleElement("Element5b"));
        gridOld[Coord<2>(8, 2)] = e;

        Region<2> region;
        region << CoordBox<2>(Coord<2>(), dim);
        UpdateFunctor<SimpleContainer>()(region, Coord<2>(), Coord<2>(), gridOld, &gridNew, 0);

        LogType expectedLog;
        expectedLog << std::make_pair("Node10", "Node10")
                    << std::make_pair("Node10", "Node11")
                    << std::make_pair("Node10", "Element5b");

        expectedLog << std::make_pair("Node11", "Node10")
                    << std::make_pair("Node11", "Node11")
                    << std::make_pair("Node11", "Element5b");

        expectedLog << std::make_pair("Element5b", "Node10")
                    << std::make_pair("Element5b", "Node11")
                    << std::make_pair("Element5b", "Element5b");

        expectedLog << std::make_pair("Node1",  "Node1")
                    << std::make_pair("Node1",  "Node5a")
                    << std::make_pair("Node1",  "Node6")
                    << std::make_pair("Node1",  "Element1")
                    << std::make_pair("Node1",  "Element7")
                    << std::make_pair("Node1",  "Element9");

        expectedLog << std::make_pair("Node5a", "Node1")
                    << std::make_pair("Node5a", "Node5a")
                    << std::make_pair("Node5a", "Node6")
                    << std::make_pair("Node5a", "Element1")
                    << std::make_pair("Node5a", "Element7")
                    << std::make_pair("Node5a", "Element9");

        expectedLog << std::make_pair("Node6",  "Node1")
                    << std::make_pair("Node6",  "Node5a")
                    << std::make_pair("Node6",  "Node6")
                    << std::make_pair("Node6",  "Element1")
                    << std::make_pair("Node6",  "Element7")
                    << std::make_pair("Node6",  "Element9");

        expectedLog << std::make_pair("Element1", "Node1")
                    << std::make_pair("Element1", "Node5a")
                    << std::make_pair("Element1", "Node6")
                    << std::make_pair("Element1", "Element1")
                    << std::make_pair("Element1", "Element7")
                    << std::make_pair("Element1", "Element9");

        expectedLog << std::make_pair("Element7", "Node1")
                    << std::make_pair("Element7", "Node5a")
                    << std::make_pair("Element7", "Node6")
                    << std::make_pair("Element7", "Element1")
                    << std::make_pair("Element7", "Element7")
                    << std::make_pair("Element7", "Element9");

        expectedLog << std::make_pair("Element9", "Node1")
                    << std::make_pair("Element9", "Node5a")
                    << std::make_pair("Element9", "Node6")
                    << std::make_pair("Element9", "Element1")
                    << std::make_pair("Element9", "Element7")
                    << std::make_pair("Element9", "Element9");

        TS_ASSERT_EQUALS(expectedLog, multiContainerCellTestLog);
    }

    void testBoxCell()
    {
        Coord<2> dim(10, 5);
        Grid<AnotherSimpleContainer> gridOld(dim);
        Grid<AnotherSimpleContainer> gridNew(dim);

        gridOld[Coord<2>(0, 0)].elements.insert(1024, SimpleElement("kiloblaster"));

        for (int y = 0; y < dim.y(); ++y) {
            for (int x = 0; x < dim.x(); ++x) {
                gridOld[Coord<2>(x, y)].particles << SimpleParticle(x + 0.5, y + 0.5);
            }
        }

        Region<2> region;
        region << CoordBox<2>(Coord<2>(), dim);
        UpdateFunctor<AnotherSimpleContainer>()(
            region,
            Coord<2>(),
            Coord<2>(),
            gridOld,
            &gridNew,
            12345);

        for (int y = 0; y < dim.y(); ++y) {
            for (int x = 0; x < dim.x(); ++x) {
                int expectedNeighbors = 9;
                if (y == 0) {
                    expectedNeighbors -= 3;
                }
                if (y == (dim.y() - 1)) {
                    expectedNeighbors -= 3;
                }
                if (x == 0) {
                    expectedNeighbors -= 3;
                }
                if (x == (dim.x() - 1)) {
                    expectedNeighbors -= 3;
                }
                // need to correct for doubly removed corner
                if (expectedNeighbors == 3) {
                    expectedNeighbors += 1;
                }

                int expectedElements = 0;
                if ((x <= 1) && (y <= 1)) {
                    expectedElements = 1;
                }

                TS_ASSERT_EQUALS(gridNew[Coord<2>(x, y)].particles[0].seenNeighbors, expectedNeighbors);
                TS_ASSERT_EQUALS(gridNew[Coord<2>(x, y)].particles[0].seenElements,  expectedElements);
            }
        }
    }

    void testSpawn()
    {
        Coord<3> dim(10, 5, 7);
        Region<3> region;
        region << CoordBox<3>(Coord<3>(), dim);
        FloatCoord<3> boxDim(2.5, 3.5, 4.5);

        Grid<CellWithSpawningParticles, Topologies::Torus<3>::Topology> gridOld(dim);
        Grid<CellWithSpawningParticles, Topologies::Torus<3>::Topology> gridNew(dim);

        for (Region<3>::Iterator i = region.begin(); i != region.end(); ++i) {
            FloatCoord<3> boxOrigin = boxDim.scale(*i);
            FloatCoord<3> particleCenter = boxOrigin + boxDim * 0.5;

            CellWithSpawningParticles cell;
            cell.red  = BoxCell<FixedArray<SpawningParticleRed,   16> >(boxOrigin, boxDim);
            cell.blue = BoxCell<FixedArray<SpawningParticleBlue, 221> >(boxOrigin, boxDim);

            cell.red  << SpawningParticleRed( particleCenter, i->x(), i->y());
            cell.blue << SpawningParticleBlue(particleCenter, i->z(), i->prod());

            gridOld[*i] = cell;
        }

        UpdateFunctor<CellWithSpawningParticles>()(
            region,
            Coord<3>(),
            Coord<3>(),
            gridOld,
            &gridNew,
            0);

        for (Region<3>::Iterator i = region.begin(); i != region.end(); ++i) {
            int expectedRedParticles  = 1 + i->x() + i->z();
            int expectedBlueParticles = 1 + i->y() + i->prod();

            TS_ASSERT_EQUALS(gridNew[*i].red.size(),  expectedRedParticles);
            TS_ASSERT_EQUALS(gridNew[*i].blue.size(), expectedBlueParticles);
        }
    }

};

}
