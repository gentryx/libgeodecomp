#include <libgeodecomp.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>


using namespace LibGeoDecomp;

const double CONTAINER_DIM = 3.0;
const double SPHERE_RADIUS = 0.6;
const double BOUNDARY_DIM  = 3.0;
const double DELTA_T = 0.01;

class BaseContainer;

class MyAPI :
    public APITraits::HasCubeTopology<3>,
    public APITraits::HasOpaqueMPIDataType<BaseContainer>,
    public APITraits::HasStencil<Stencils::Moore<3, 1> >,
    public APITraits::HasNanoSteps<2>
{};

class Sphere;

class Boundary
{
public:
    friend class Sphere;
    friend class GasWriter;

    typedef MyAPI API;

    explicit Boundary(
        const FloatCoord<3>& myCenter = FloatCoord<3>(),
        const FloatCoord<3>& myNormal = FloatCoord<3>()) :
        center(myCenter),
        normal(myNormal),
        glow(0)
    {}

    template<typename HOOD>
    void update(const HOOD& hood, const int nanoStep)
    {
        glow -= DELTA_T * 0.04;
        if (glow < 0) {
            glow = 0;
        }

        checkForCollisions(hood.spheres.begin(), hood.spheres.end());
    }

    template<typename ITERATOR>
    void checkForCollisions(const ITERATOR& begin, const ITERATOR& end)
    {
        for (ITERATOR i = begin; i != end; ++i) {
            if (i->force(*this) != FloatCoord<3>()) {
                glow = 1;
            }
        }
    }

    const FloatCoord<3> getPos() const
    {
        return center;
    }

private:
    FloatCoord<3> center;
    FloatCoord<3> normal;
    double glow;
};


class Sphere
{
public:
    friend class GasWriter;
    friend class Container;

    typedef MyAPI API;

    explicit Sphere(
        const FloatCoord<3>& myPos = FloatCoord<3>(),
        const FloatCoord<3>& myVel = FloatCoord<3>()) :
        pos(myPos),
        vel(myVel)
    {}

    template<typename HOOD>
    void update(const HOOD& hood, const int nanoStep)
    {
        addForces(hood.spheres.begin(),    hood.spheres.end());
        addForces(hood.boundaries.begin(), hood.boundaries.end());

        pos += vel * DELTA_T;
    }

    template<typename ITERATOR>
    void addForces(const ITERATOR& begin, const ITERATOR& end)
    {
        for (ITERATOR i = begin; i != end; ++i) {
            vel += force(*i) * DELTA_T;
        }
    }

    const FloatCoord<3> getPos() const
    {
        return pos;
    }

    FloatCoord<3> force(const Sphere& other) const
    {
        FloatCoord<3> ret;
        FloatCoord<3> delta = pos - other.pos;
        double distance = delta.length();

        if (distance == 0) {
            return ret;
        }

        if (distance < (2.0 * SPHERE_RADIUS)) {
            double scale = (SPHERE_RADIUS * SPHERE_RADIUS) / distance / distance;
            ret = delta * scale;
        }

        return ret;
    }

    FloatCoord<3> force(const Boundary& other) const
    {
        FloatCoord<3> ret;
        FloatCoord<3> delta = pos - other.center;
        double distance =
            delta[0] * other.normal[0] +
            delta[1] * other.normal[1] +
            delta[2] * other.normal[2];

        if (distance < SPHERE_RADIUS) {
            FloatCoord<3> planar = (pos - other.normal * distance) - other.center;
            if ((fabs(planar[0]) < (BOUNDARY_DIM * 0.5)) &&
                (fabs(planar[1]) < (BOUNDARY_DIM * 0.5)) &&
                (fabs(planar[2]) < (BOUNDARY_DIM * 0.5))) {
                double scale = (SPHERE_RADIUS * SPHERE_RADIUS) / distance / distance;
                ret = other.normal * scale;
            }
        }

        return ret;
    }

private:
    FloatCoord<3> pos;
    FloatCoord<3> vel;
};

typedef BoxCell<FixedArray<Sphere,   30> > SpheresContainer;
typedef BoxCell<FixedArray<Boundary, 30> > BoundariesContainer;

DECLARE_MULTI_CONTAINER_CELL(
    BaseContainer,
    Sphere,
    ((SpheresContainer)(spheres))
    ((BoundariesContainer)(boundaries)) )

class Container : public BaseContainer
{
public:
    friend class GasWriter;

    typedef MyAPI API;

    explicit Container(
        const FloatCoord<3>& origin = FloatCoord<3>(),
        const FloatCoord<3>& dimensions = FloatCoord<3>())
    {
        spheres    = BoxCell<FixedArray<Sphere,   30> >(origin, dimensions);
        boundaries = BoxCell<FixedArray<Boundary, 30> >(origin, dimensions);
    }

};

class GasWriter : public Clonable<Writer<Container>, GasWriter>
{
public:
    typedef APITraits::SelectTopology<Container>::Value Topology;
    static const int DIM = Topology::DIM;
    typedef GridBase<Container, DIM> GridType;

    explicit GasWriter(
        const std::string& prefix,
        const unsigned period = 1) :
        Clonable<Writer<Container>, GasWriter>(prefix, period)
    {}

    virtual void stepFinished(const GridType& grid, unsigned step, WriterEvent event)
    {
        if ((event == WRITER_STEP_FINISHED) && (step % period != 0)) {
            return;
        }

        std::stringstream filename;
        filename << prefix << "_" << std::setfill('0') << std::setw(6)
                 << step << ".pov";
        std::ofstream file(filename.str().c_str());

        file << "#include \"colors.inc\"\n"
             << "background { color Black }\n"
             << "plane {\n"
             << "  <0, 1, 0>, 0 \n"
             << "  pigment { color Red }\n"
             << "} \n"
             << "camera { \n"
             << "  location <-10, 25, -30> \n"
             << "  look_at  <5, 10,  20> \n"
             << "  right 16/9*x\n"
             << "} \n"
             << "light_source { <20, 30, -30> color White}\n\n";

        CoordBox<3> box = grid.boundingBox();
        for (CoordBox<3>::Iterator j = box.begin(); j != box.end(); ++j) {
            const Container& container = grid.get(*j);

            for (std::size_t i = 0; i < container.spheres.size(); ++i) {
                file << sphereToPOV(container.spheres[i]);
            }

            for (std::size_t i = 0; i < container.boundaries.size(); ++i) {
                file << boundaryToPOV(container.boundaries[i]);
            }
        }
    }

private:
    using Writer<Container>::period;
    using Writer<Container>::prefix;

    std::string sphereToPOV(const Sphere& ball)
    {
        std::stringstream buf;
        buf << "sphere {\n"
            << "  <" << ball.pos[0] << ", " << ball.pos[1] << ", "
            << ball.pos[2] << ">, " << SPHERE_RADIUS << "\n"
            << "  texture {\n"
            << "    pigment {color White}\n"
            << "    finish {phong 0.9 metallic}\n"
            << "  }\n"
            << "}\n";
        return buf.str();
    }

    std::string boundaryToPOV(const Boundary& tile)
    {
        if (tile.glow == 0) {
            return "";
        }

        FloatCoord<3> diag(BOUNDARY_DIM * 0.5,
                           BOUNDARY_DIM * 0.5,
                           BOUNDARY_DIM * 0.5);
        FloatCoord<3> corner1 = tile.center - diag;
        FloatCoord<3> corner2 = tile.center + diag;
        double factor = 0.4;
        if (tile.normal.sum() < 0) {
            factor = -factor;
        }
        corner1 += tile.normal * BOUNDARY_DIM * factor;
        corner2 -= tile.normal * BOUNDARY_DIM * factor;

        double transmit = 1.0 - tile.glow * 0.3;
        double ior = 1.0 + tile.glow * 0.5;

        std::stringstream buf;
        buf << "box {\n"
            << "  <" <<  corner1[0]
            << ", "  <<  corner1[1]
            << ", "  <<  corner1[2]
            << ">, <" << corner2[0]
            << ", "  <<  corner2[1]
            << ", "  <<  corner2[2]
            << ">\n"
            << "  texture {\n"
            << "    pigment {color Blue transmit " << transmit << "}\n"
            << "    finish {phong 0.9 metallic}\n"
            << "  }\n"
            << "  interior {ior " << ior << "}\n"
            << "}\n";
        return buf.str();
    }
};

class GasInitializer : public SimpleInitializer<Container>
{
public:
    GasInitializer(const Coord<3>& dimensions, unsigned steps) :
        SimpleInitializer<Container>(dimensions, steps)
    {}

    virtual void grid(GridBase<Container, 3> *target)
    {
        CoordBox<3> box = target->boundingBox();

        for (CoordBox<3>::Iterator j = box.begin(); j != box.end(); ++j) {
            // use the Initializer's seeding feature to ensure
            // consistent initialization of overlapping cells on all
            // nodes.
            seedRNG(*j);
            double pseudo_rand1 = (Random::genDouble() - 0.5) / 16.0;
            double pseudo_rand2 = (Random::genDouble() - 0.5) / 16.0;
            double pseudo_rand3 = (Random::genDouble() - 0.5) / 16.0;

            FloatCoord<3> center(
                (j->x() + 0.5) * CONTAINER_DIM,
                (j->y() + 0.5) * CONTAINER_DIM,
                (j->z() + 0.5) * CONTAINER_DIM);

            Container container(FloatCoord<3>(*j) * CONTAINER_DIM,
                                FloatCoord<3>(CONTAINER_DIM, CONTAINER_DIM, CONTAINER_DIM));

            container.spheres << Sphere(
                center,
                FloatCoord<3>(
                    pseudo_rand1,
                    pseudo_rand2,
                    pseudo_rand3));

            // Bounday elements should align centered to the outsides
            // of the cells. I prefer this slightly complicated code
            // to overly reduncant code.

            // left boundary
            if (j->x() == 0) {
                addBoundary(&container, center, FloatCoord<3>(1, 0, 0));
            }

            // right boundary
            if (j->x() == (box.dimensions[0] - 1))
                addBoundary(&container, center, FloatCoord<3>(-1, 0, 0));

            // lower boundary
            if (j->y() == 0) {
                addBoundary(&container, center, FloatCoord<3>(0, 1, 0));
            }

            // upper boundary
            if (j->y() == (box.dimensions[1] - 1)) {
                addBoundary(&container, center, FloatCoord<3>(0, -1, 0));
            }

            // front boundary
            if (j->z() == 0) {
                addBoundary(&container, center, FloatCoord<3>(0, 0, 1));
            }

            // rear boundary
            if (j->z() == (box.dimensions[2] - 1)) {
                addBoundary(&container, center, FloatCoord<3>(0, 0, -1));
            }

            target->set(*j, container);
        }
    }

private:
    void addBoundary(
        Container *container,
        const FloatCoord<3>& containerCenter,
        const FloatCoord<3>& normal)
    {
        FloatCoord<3> boundaryCenter = containerCenter - normal * (CONTAINER_DIM * 0.4);
        container->boundaries << Boundary(boundaryCenter, normal);
    }

};

int main(int argc, char **argv)
{
    SerialSimulator<Container> sim(
        new GasInitializer(
            Coord<3>(10, 10, 10),
            80000));
    sim.addWriter(
        new GasWriter(
            "sim",
            200));

    sim.run();

    return 0;
}
