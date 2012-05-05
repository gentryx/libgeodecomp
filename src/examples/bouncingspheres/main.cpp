#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/misc/floatcoord.h>
#include <libgeodecomp/parallelization/serialsimulator.h>

using namespace LibGeoDecomp;

const int CONTAINER_SIZE = 30;
const double CONTAINER_DIM = 3.0;
const double SPHERE_RADIUS = 0.6;
const double BOUNDARY_DIM  = 3.0;
const double DELTA_T = 0.01;

class Sphere;

class Boundary
{
public:
    friend class Sphere;
    friend class GasWriter;
    
    Boundary(
        const FloatCoord<3>& myCenter = FloatCoord<3>(), 
        const FloatCoord<3>& myNormal = FloatCoord<3>()) :
        center(myCenter),
        normal(myNormal),
        glow(0)
    {}

    void update(
        const Sphere **neighborSpheres, 
        const int *numSpheres);

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

    Sphere(
        const int& myID = 0,
        const FloatCoord<3>& myPos = FloatCoord<3>(), 
        const FloatCoord<3>& myVel = FloatCoord<3>()) :
        id(myID),
        pos(myPos),
        vel(myVel)
    {}

    void update(
        const Coord<3>& parentOrigin, 
        const Sphere **neighborSpheres, 
        const int *numSpheres,
        const Boundary **neighborBoundaries,
        const int *numBoundaries)
    {
        for (int i = 0; i < 27; ++i) {
            for (int j = 0; j < numSpheres[i]; ++j) 
                if (neighborSpheres[i][j].id != id)

                    vel += force(neighborSpheres[i][j]) * DELTA_T;
            for (int j = 0; j < numBoundaries[i]; ++j)
                vel += force(neighborBoundaries[i][j]) * DELTA_T;
        }
        pos += vel * DELTA_T;

        // need to determine to which container to move next
        for (int d = 0; d < 3; ++d) {
            double val = 0;
            if (pos[d] < parentOrigin[d]) {
                val = -1;
            }
            if (pos[d] >= (parentOrigin[d] + CONTAINER_DIM)) {
                val = 1;
            }
            targetContainer[d] = val;
        }
    }

    FloatCoord<3> force(const Sphere& other) const
    {
        FloatCoord<3> ret;
        FloatCoord<3> delta = pos - other.pos;
        double distance = delta.length();

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
            if ((abs(planar[0]) < (BOUNDARY_DIM * 0.5)) &&
                (abs(planar[1]) < (BOUNDARY_DIM * 0.5)) &&
                (abs(planar[2]) < (BOUNDARY_DIM * 0.5))) {
                double scale = (SPHERE_RADIUS * SPHERE_RADIUS) / distance / distance;
                ret = other.normal * scale;
            }
        }

        return ret;
    }

private:
    int id;
    FloatCoord<3> pos;
    FloatCoord<3> vel;
    Coord<3> targetContainer;
    double col;
};

class Container
{
public:
    friend class GasWriter;

    typedef Topologies::Cube<3>::Topology Topology;

    Container(const Coord<3>& myOrigin = Coord<3>()) :
        origin(myOrigin),
        numSpheres(0),
        numBoundaries(0)
    {}

    static unsigned nanoSteps()
    {
        return 2;
    }

    template<typename COORD_MAP>
    void update(const COORD_MAP& neighborhood, const unsigned& nanoStep)
    {
        *this = neighborhood[Coord<3>()];

        if (nanoStep == 0) {
            updateCargo(neighborhood);
        } else {
            moveSpheres(neighborhood);
        }
    }

    void addSphere(const Sphere& s)
    {
        if (numSpheres >= CONTAINER_SIZE)
            throw std::logic_error("too many spheres");
        spheres[numSpheres] = s;
        ++numSpheres;
    }

    void addBoundary(const Boundary& b)
    {
        if (numBoundaries >= CONTAINER_SIZE)
            throw std::logic_error("too many boundaries");
        boundaries[numBoundaries] = b;
        ++numBoundaries;
    }

private:
    Sphere spheres[CONTAINER_SIZE];
    Boundary boundaries[CONTAINER_SIZE];
    Coord<3> origin;
    int numSpheres;
    int numBoundaries;

    template<typename COORD_MAP>
    void updateCargo(const COORD_MAP& neighborhood)
    {
        const Sphere *hoodS[27];
        const Boundary *hoodB[27];
        int numS[27];
        int numB[27];
        CoordBox<3> box(Coord<3>::diagonal(-1), Coord<3>::diagonal(3));
        CoordBoxSequence<3> s = box.sequence();
        int i = 0;

        while (s.hasNext()) {
            Coord<3> c = s.next();
            hoodS[i] = neighborhood[c].spheres;
            hoodB[i] = neighborhood[c].boundaries;
            numS[i] = neighborhood[c].numSpheres;
            numB[i] = neighborhood[c].numBoundaries;
            ++i;
        }

        for (int i = 0; i < numSpheres; ++i)
            spheres[i].update(origin, hoodS, numS, hoodB, numB);
        for (int i = 0; i < numBoundaries; ++i)
            boundaries[i].update(hoodS, numS);
    }

    template<typename COORD_MAP>
    void moveSpheres(const COORD_MAP& neighborhood)
    {
        CoordBox<3> box(Coord<3>::diagonal(-1), Coord<3>::diagonal(3));
        CoordBoxSequence<3> s = box.sequence();
        numSpheres = 0;

        while (s.hasNext()) {
            Coord<3> c = s.next();
            const Container& other = neighborhood[c];
            for (int i = 0; i < other.numSpheres; ++i) {
                if (other.spheres[i].targetContainer == -c) {
                    spheres[numSpheres] = other.spheres[i];
                    ++numSpheres;
                }
            }
        }
        
    }
    
};

void Boundary::update(
    const Sphere **neighborSpheres, 
    const int *numSpheres)
{
    glow -= DELTA_T * 0.04;

    if (glow < 0)
        glow = 0;

    for (int i = 0; i < 27; ++i) 
        for (int j = 0; j < numSpheres[i]; ++j) 
            if (neighborSpheres[i][j].force(*this) != FloatCoord<3>())
                glow = 1;
}

class GasWriter : public Writer<Container>
{
public:
    GasWriter(
        const std::string& _prefix,
        MonolithicSimulator<Container> *_sim,
        const unsigned _period = 1) :
        Writer<Container>(_prefix, _sim, _period)
    {}

    virtual void initialized()
    {
        writeStep();
    }

    virtual void stepFinished()
    {        
        writeStep();
    }

    virtual void allDone()
    {
        writeStep();
    }

private:
    void writeStep()
    {
        if (this->sim->getStep() % this->period != 0)
            return;

        std::stringstream filename;
        filename << this->prefix << "_" << std::setfill('0') << std::setw(6) 
                 << this->sim->getStep() << ".pov";
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

        CoordBox<3> box = this->sim->getGrid()->boundingBox();
        CoordBoxSequence<3> s = box.sequence();
        while (s.hasNext()) {
            Coord<3> c = s.next();
            const Container& container = (*this->sim->getGrid())[c];

            for (int i = 0; i < container.numSpheres; ++i) 
                file << sphereToPOV(container.spheres[i]);
            
            for (int i = 0; i < container.numBoundaries; ++i) 
                file << boundaryToPOV(container.boundaries[i]);
        }
    }

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
        if (tile.glow == 0)
            return "";

        FloatCoord<3> diag(BOUNDARY_DIM * 0.5,
                           BOUNDARY_DIM * 0.5,
                           BOUNDARY_DIM * 0.5);
        FloatCoord<3> corner1 = tile.center - diag;
        FloatCoord<3> corner2 = tile.center + diag;
        double factor = 0.4;
        if (tile.normal.sum() < 0) 
            factor = -factor;
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
    GasInitializer(const Coord<3>& dimensions, const unsigned& steps) : 
        SimpleInitializer<Container>(dimensions, steps)
    {}

    virtual void grid(GridBase<Container, 3> *target) 
    {
        CoordBox<3> box = target->boundingBox();

        CoordBoxSequence<3> s = box.sequence();
        while (s.hasNext()) {
            Coord<3> c = s.next();
            // We use bad pseudo random numbers as we need to ensure
            // that all cells get initialized the same way on all MPI
            // nodes. This would be hard with a stateful pseudo random
            // number generator.
            double pseudo_rand1 = (c.sum() * 11 % 16 - 8) / 256.0;
            double pseudo_rand2 = (c.sum() * 31 % 16 - 8) / 256.0;
            double pseudo_rand3 = (c.sum() * 41 % 16 - 8) / 256.0;
            
            FloatCoord<3> center(
                (c.x() + 0.5) * CONTAINER_DIM, 
                (c.y() + 0.5) * CONTAINER_DIM, 
                (c.z() + 0.5) * CONTAINER_DIM);

            Container container(c * CONTAINER_DIM);

            container.addSphere(
                Sphere(
                    c * Coord<3>(1, 100, 10000),
                    center,
                    FloatCoord<3>(
                        pseudo_rand1,
                        pseudo_rand2,
                        pseudo_rand3)));
            
            // Bounday elements should align centered to the outsides
            // of the cells. I prefer this slightly complicated code
            // to overly reduncant code.
            
            // left boundary
            if (c[0] == 0)
                addBoundary(&container, center, FloatCoord<3>(1, 0, 0));

            // right boundary
            if (c[0] == (box.dimensions[0] - 1))
                addBoundary(&container, center, FloatCoord<3>(-1, 0, 0));

            // lower boundary
            if (c[1] == 0)
                addBoundary(&container, center, FloatCoord<3>(0, 1, 0));

            // upper boundary
            if (c[1] == (box.dimensions[1] - 1))
                addBoundary(&container, center, FloatCoord<3>(0, -1, 0));

            // front boundary
            if (c[2] == 0)
                addBoundary(&container, center, FloatCoord<3>(0, 0, 1));

            // rear boundary
            if (c[2] == (box.dimensions[2] - 1))
                addBoundary(&container, center, FloatCoord<3>(0, 0, -1));

            target->at(c) = container;
        }
    }

private:
    void addBoundary(
        Container *container,
        const FloatCoord<3>& containerCenter, 
        const FloatCoord<3>& normal)
    {
        FloatCoord<3> boundaryCenter = containerCenter - normal * (CONTAINER_DIM * 0.5);
        container->addBoundary(Boundary(boundaryCenter, normal));
    }

};

int main(int argc, char **argv)
{
    SerialSimulator<Container> sim(
        new GasInitializer(
            Coord<3>(10, 10, 10), 
            40000));
    new GasWriter(
        "sim",
        &sim,
        200);
                           
    sim.run();

    return 0;
}
