#ifndef _libgeodecomp_examples_flowingcanvas_canvascell_h_
#define _libgeodecomp_examples_flowingcanvas_canvascell_h_

#include <libgeodecomp/misc/floatcoord.h>
#include <libgeodecomp/misc/topologies.h>

namespace LibGeoDecomp {

class CanvasCell
{
public:
    class Particle
    {
    public:
        Particle(const float& pos0 = 0, const float& pos1 = 0, const float& _lifetime = 1000) :
            lifetime(_lifetime)
        {
            pos[0] = pos0;
            pos[1] = pos1;
            vel[0] = 0;
            vel[1] = 0;
        }

        void update(const float& deltaT, const float& force0, const float& force1, const float& forceFactor, const float& friction)
        {
            vel[0] += deltaT * forceFactor * force0;
            vel[1] += deltaT * forceFactor * force1;
            vel[0] *= friction;
            vel[1] *= friction;
            // pos[0] += deltaT * vel[0];
            // pos[1] += deltaT * vel[1];
            --lifetime;
        }

        float pos[2];
        float vel[2];
        int lifetime;
    };

    typedef Topologies::Cube<2>::Topology Topology;
    static const int TILE_WIDTH = 4;
    static const int TILE_HEIGHT = 4;
    
    static inline unsigned nanoSteps()
    {
        return 1;
    }

    CanvasCell(
        Coord<2> _pos = Coord<2>(), 
        bool _forceSet = false,
        FloatCoord<2> _forceFixed = FloatCoord<2>()) :
        cameraLevel(0),
        numParticles(0)
    {
        pos[0] = _pos.x();
        pos[1] = _pos.y();
        forceFixed[0] = _forceFixed.c[0];
        forceFixed[1] = _forceFixed.c[1];
        forceSet = _forceSet;
    }

    template<typename COORD_MAP>
    void update(const COORD_MAP& hood, const unsigned& nanoStep)
    {
        const CanvasCell& oldSelf = hood[Coord<2>()];

        pos[0] = oldSelf.pos[0];
        pos[1] = oldSelf.pos[1];
        cameraPixel = oldSelf.cameraPixel;
        cameraLevel = oldSelf.cameraLevel;
        numParticles = oldSelf.numParticles;
        for (int i = 0; i < numParticles; ++i) {
            particles[i] = oldSelf.particles[i];
        } 
        // fixme: render particles
        // fixme: spawn particles
        // fixme: move particles to other cells
        // fixme: kill dead particles
        if (numParticles < 1) {
            particles[numParticles] = Particle(pos[0], pos[1]);
            numParticles = 1;
        }

        forceSet = oldSelf.forceSet;
        if (forceSet) {
            forceFixed[0] = oldSelf.forceFixed[0];
            forceFixed[1] = oldSelf.forceFixed[1];
        } else {
            forceFixed[0] = (hood[Coord<2>(0, -1)].forceFixed[0] +
                             hood[Coord<2>(-1, 0)].forceFixed[0] +
                             hood[Coord<2>(1,  0)].forceFixed[0] +
                             hood[Coord<2>(0,  1)].forceFixed[0]) * 0.25;
            forceFixed[1] = (hood[Coord<2>(0, -1)].forceFixed[1] +
                             hood[Coord<2>(-1, 0)].forceFixed[1] +
                             hood[Coord<2>(1,  0)].forceFixed[1] +
                             hood[Coord<2>(0,  1)].forceFixed[1]) * 0.25;
        }

        cameraLevel = (hood[Coord<2>(0, -1)].cameraLevel +
                             hood[Coord<2>(-1, 0)].cameraLevel +
                             hood[Coord<2>(1,  0)].cameraLevel +
                             hood[Coord<2>(0,  1)].cameraLevel) * 0.25;
        
        float gradientX = hood[Coord<2>(1, 0)].cameraLevel - hood[Coord<2>(-1, 0)].cameraLevel;
        float gradientY = hood[Coord<2>(0, 1)].cameraLevel - hood[Coord<2>(0, -1)].cameraLevel;
        forceVario[0] = 0;
        if ((gradientX > 0.011) || (gradientX < -0.011)) {
            forceVario[0] = 0.01 / gradientX;
        } else {
            forceVario[0] = 0;
        }

        if ((gradientY > 0.011) || (gradientY < -0.011)) {
            forceVario[1] = 0.01 / gradientY;
        } else {
            forceVario[1] = 0;
        }

        forceTotal[0] = 0.5 * (forceFixed[0] + forceVario[0]);
        forceTotal[1] = 0.5 * (forceFixed[1] + forceVario[1]);

        for (int i = 0; i < numParticles; ++i) {
            Particle& p = particles[i];
            // fixme: parameters
            p.update(1.0, forceTotal[0], forceTotal[1], 1.0, 0.99);
        }
        
//         float gradient[2];
//         gradient[0] = hood[Coord<2>(1, 0)].smoothCam - hood[Coord<2>(-1, 0)].smoothCam;
//         gradient[1] = hood[Coord<2>(0, 1)].smoothCam - hood[Coord<2>(0, -1)].smoothCam;

//         forceVario[0] = 0;
//         forceVario[1] = 0;
        
//         if ((gradient[0] * gradient[0]) > gradientCutoff) {
//             forceVario[0] = 1.0 / gradient[0];
//         }

//         if ((gradient[1] * gradient[1]) > gradientCutoff) {
//             forceVario[1] = 1.0 / gradient[1];
//         }

//         updateParticles(oldSelf.particle, 
//                         forceVario[0] + forceFixed[0],
//                         forceVario[1] + forceFixed[1]);
//         // moveParticles();
    }

    void readCam(unsigned char *frame, const float& factorX, const float& factorY, const int& width, const int& height)
    {
        int posX = pos[0] * factorX;
        int posY = pos[1] * factorY;
        
        int index = posY * width * 3 + posX * 3;
        cameraPixel = (0xff << 24) + 
            (frame[index + 0] << 16) +
            (frame[index + 1] <<  8) +
            (frame[index + 2] <<  0);

        int val = (int)frame[index + 0] + (int)frame[index + 1] + (int)frame[index + 2];
        if (val < 500) {
            cameraLevel = 1.0;
        } else {
            cameraLevel = std::max(0.0, cameraLevel - 0.05);
        }
    }

    // fixme:
// private:
    unsigned color[TILE_HEIGHT][TILE_WIDTH];
    unsigned cameraPixel;
    float pos[2];
    bool forceSet;

    float cameraLevel;
    float forceVario[2];
    float forceFixed[2];
    float forceTotal[2];
    int numParticles;
    Particle particles[20];  

    void updateParticles()
    {}

    void moveParticles()
    {}
};

}

#endif
