#ifndef _libgeodecomp_examples_flowingcanvas_canvascell_h_
#define _libgeodecomp_examples_flowingcanvas_canvascell_h_

#include <stdio.h>
#include <libgeodecomp/misc/cellapitraits.h>
#include <libgeodecomp/misc/floatcoord.h>
#include <libgeodecomp/misc/stencils.h>
#include <libgeodecomp/misc/topologies.h>

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

namespace LibGeoDecomp {

class CanvasCell
{
    friend class CanvasWriter;
public:
    class Particle
    {
    public:
        __host__ __device__
        Particle(const float& pos0 = 0, const float& pos1 = 0, const unsigned& _color = 0, const int& _lifetime = 100) :
            lifetime(_lifetime),
            color(_color)
        {
            pos[0] = pos0;
            pos[1] = pos1;
            vel[0] = 0;
            vel[1] = 0;
        }

        __host__ __device__
        void update(const float& deltaT, const float& force0, const float& force1, const float& forceFactor, const float& friction)
        {
            vel[0] += deltaT * forceFactor * force0;
            vel[1] += deltaT * forceFactor * force1;
            vel[0] *= friction;
            vel[1] *= friction;
            pos[0] += deltaT * vel[0];
            pos[1] += deltaT * vel[1];
            --lifetime;
        }

        float pos[2];
        float vel[2];
        int lifetime;
        unsigned color;
    };

    static const int MAX_PARTICLES = 10;
    static const int MAX_SPAWN_COUNTDOWN = 60;

    typedef Stencils::Moore<2, 1> Stencil;
    typedef Topologies::Cube<2>::Topology Topology;
    class API : public CellAPITraits::Base
    {};

    __host__ __device__
    static inline unsigned nanoSteps()
    {
        return 4;
    }

    CanvasCell(
        const unsigned& _originalPixel = 0,
        const Coord<2>& _pos = Coord<2>(), 
        const bool& _forceSet = false,
        const FloatCoord<2>& _forceFixed = FloatCoord<2>(),
        const int& _spawnCountdown = 0) :
        originalPixel(_originalPixel),
        spawnCountdown(_spawnCountdown),
        cameraLevel(0),
        numParticles(0)
    {
        pos[0] = _pos.x();
        pos[1] = _pos.y();
        forceFixed[0] = _forceFixed[0];
        forceFixed[1] = _forceFixed[1];
        forceSet = _forceSet;
    }

    __host__ __device__
    void update(const CanvasCell *up, const CanvasCell *same, const CanvasCell *down, const unsigned& nanoStep)
    {
        if (nanoStep == (nanoSteps() - 1)) {
            // fixme: can we avoid this?
            *this = *same;
            moveParticles(up, same, down);
            return;
        }

        updateForces(up, same, down, nanoStep);

        if (nanoStep == (nanoSteps() - 2)) {
            spawnParticles();
            updateParticles();
        }
    }

    template<typename COORD_MAP>
    void update(const COORD_MAP& hood, const unsigned& nanoStep)
    {
        update(&hood[Coord<2>(0, -1)], &hood[Coord<2>(0, 0)], &hood[Coord<2>(0, 1)], nanoStep);
    }


    __host__ __device__
    void readCam(const unsigned char& r, const unsigned char& g, const unsigned char& b)
    {
        cameraPixel = (0xff << 24) + (r << 16) + (g <<  8) + (b <<  0);

        int val = (int)r + (int)g + (int)b;
        if (val < 250) {
            cameraLevel = 1.0;
        } else {
            cameraLevel = max(0.0, cameraLevel - 0.02);
        }
    }

    // fixme:
private:
    unsigned originalPixel;
    unsigned cameraPixel;
    unsigned spawnCountdown;
    float pos[2];
    bool forceSet;

    float cameraLevel;
    float forceVario[2];
    float forceFixed[2];
    float forceTotal[2];
    int numParticles;
    Particle particles[MAX_PARTICLES];  

    /**
     * Transition particles between neighboring cells
     */
    __host__ __device__
    void moveParticles(const CanvasCell *up, const CanvasCell *same, const CanvasCell *down)
    {
        numParticles = 0;
        for (int x = -1; x < 2; ++x) {
            addParticles(up[x]);
            addParticles(same[x]);
            addParticles(down[x]);
        }
    }

    __host__ __device__
    void addParticles(const CanvasCell& cell)
    {
        for (int i = 0; i < cell.numParticles; ++i) {
            const Particle& particle = cell.particles[i];
            // fixme: refactor via "extract method" for readability
            if ((((int)pos[0]) == ((int)particle.pos[0])) &&
                (((int)pos[1]) == ((int)particle.pos[1])) &&
                (particle.lifetime > 0)) {
                if (numParticles < MAX_PARTICLES) {
                    particles[numParticles] = particle;
                    ++numParticles; 
                } else {
                    printf("uhoh\n");
                }
            }
        }
    }

    __host__ __device__
    void spawnParticles()
    {
        // fixme: render particles
        // fixme: spawn particles
        // fixme: move particles to other cells

        if (spawnCountdown <= 0) {
            spawnCountdown = MAX_SPAWN_COUNTDOWN;

            if ((numParticles < 1) && ((int)pos[0] % 5 == 0) && ((int)pos[1] % 5 == 0)) {
                particles[numParticles] = Particle(pos[0], pos[1], originalPixel);
                numParticles = 1;
            }
            
        } else {
            --spawnCountdown;
        }
    }

    __host__ __device__
    void updateParticles()
    {
        
        for (int i = 0; i < numParticles; ++i) {
            Particle& p = particles[i];
            // fixme: parameters
            p.update(0.5, forceTotal[0], forceTotal[1], 1.0, 0.9);
        }
    }

    __host__ __device__
    void updateForces(const CanvasCell *up, const CanvasCell *same, const CanvasCell *down, const unsigned& nanoStep)
    {
        const CanvasCell& oldSelf = *same;
        
        spawnCountdown = oldSelf.spawnCountdown;
        pos[0] = oldSelf.pos[0];
        pos[1] = oldSelf.pos[1];
        originalPixel = oldSelf.originalPixel;
        cameraPixel = oldSelf.cameraPixel;
        cameraLevel = oldSelf.cameraLevel;
        numParticles = oldSelf.numParticles;
        for (int i = 0; i < numParticles; ++i) {
            particles[i] = oldSelf.particles[i];
        } 

        forceSet = oldSelf.forceSet;
        if (forceSet) {
            forceFixed[0] = oldSelf.forceFixed[0];
            forceFixed[1] = oldSelf.forceFixed[1];
        } else {
            forceFixed[0] = (up[0].forceFixed[0] +
                             same[-1].forceFixed[0] +
                             same[1].forceFixed[0] +
                             down[0].forceFixed[0]) * 0.25;
            forceFixed[1] = (up[0].forceFixed[1] +
                             same[-1].forceFixed[1] +
                             same[1].forceFixed[1] +
                             down[0].forceFixed[1]) * 0.25;
        }

        cameraLevel = (up[0].cameraLevel +
                             same[-1].cameraLevel +
                             same[1].cameraLevel +
                             down[0].cameraLevel) * 0.25;
        
        float gradientX = same[1].cameraLevel - same[-1].cameraLevel;
        float gradientY = down[0].cameraLevel - up[0].cameraLevel;
        forceVario[0] = 0;

        float gradientLimit = 0.001;
        
        if ((gradientX > gradientLimit) || (gradientX < -gradientLimit)) {
            forceVario[0] = min(1.0, 0.01 / gradientX);
        } else {
            forceVario[0] = 0;
        }

        if ((gradientY > gradientLimit) || (gradientY < -gradientLimit)) {
            forceVario[1] = min(1.0, 0.01 / gradientY);
        } else {
            forceVario[1] = 0;
        }

        forceTotal[0] = 0.8 * forceFixed[0] + 0.2 * forceVario[0];
        forceTotal[1] = 0.8 * forceFixed[1] + 0.2 * forceVario[1];
    }

    __host__ __device__
    float max(const float& a, const float& b)
    {
        return a > b ? a : b;
    }

    __host__ __device__
    float min(const float& a, const float& b)
    {
        return a < b ? a : b;
    }
};

}

#endif
