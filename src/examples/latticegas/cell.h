#ifndef _libgeodecomp_examples_latticegas_cell_h_
#define _libgeodecomp_examples_latticegas_cell_h_

#include <iostream>
#include <sstream>
#include <libgeodecomp/examples/latticegas/simparams.h>

class Cell
{
public:

    enum Position { 
        UL, // upper left
        UR, // upper right
        L,  // left
        C,  // center
        R,  // right
        LL, // lower left
        LR  // lower right
    };

    enum State {
        liquid,
        solid,
        slip,
        source,
        drain
    };

#ifndef __device__
#define __device__
#endif

#ifndef __host__
#define __host__
#endif

    __device__ __host__ static int simpleRand(int i) {
        return i * 69069 + 1327217885;
    }

    class Pattern
    {
    public:
        Pattern() :
            flowState(0)
        {
            // default: particle moves back to where it came from (reflection)
            for (int i = 0; i < 7; ++i) {
                destinations[i] = (Position)i;
            }
        }

        /**
         * particle at position src will move to position dst
         */
        void operator()(const Position& src, const Position& dst)
        {
            flowState |= 1 << src;
            int cur = -1;
            for (int i = 0; i < 7; ++i)
                if (destinations[i] == dst)
                    cur = i;
            Position buf = destinations[src];
            destinations[src] = dst;
            destinations[cur] = buf;
        }

        Pattern rotate(unsigned angle) const
        {
            if (angle == 0)
                return *this;

            Pattern p;
            for (int i = 0; i < 7; ++i)
                if (isSet(i))
                    p(successor((Position)i), successor(destinations[i]));
            return p.rotate(angle - 1);
        }

        bool isSet(int i) const
        {
            return (flowState >> i) & 1;
        }

        const Position& getDest(int i) const
        {
            return destinations[i];
        }

        static std::string posToString(const Position& pos)
        {
            switch (pos) {
            case UL:
                return "UL";
            case UR:
                return "UR";
            case L:
                return "L";
            case R:
                return "R";
            case LL:
                return "LL";
            case LR:
                return "LR";
            default:
                return "C";
            }            
        }

        static Position successor(const Position& pos)
        {
            switch (pos) {
            case UL:
                return UR;
            case UR:
                return R;
            case L:
                return UL;
            case R:
                return LR;
            case LL:
                return L;
            case LR:
                return LL;
            default:
                return C;
            }
        }

        int getFlowState() const
        {
            return flowState;
        }
        
        std::string toString() const
        {
            std::ostringstream buf;
            buf << "(flow: " << flowState << ", ";
            for (int i = 0; i < 7; ++i) {
                if (isSet(i)) {
                    buf << posToString((Position)i) << "->" << posToString(destinations[i]) << ", ";
                }
            }
            buf << ")";
            return buf.str();
        }

    private:
        int flowState;
        Position destinations[7];
    };
    
    inline Cell(const State& newState=liquid, const int& k=-1, const int& val=1) :
        state(newState)
    {
        for (int i = 0; i < 7; ++i)
            particles[i] = 0;
        if (k >= 0)
            particles[k] = val;
    }

    __device__ __host__ inline void swap(char& a, char& b)
    {
        char buf = a;
        a = b;
        b = buf;
    }
    
    __device__ __host__ inline void update(
        SimParams *simParams,
        const int& t,
        const char& oldState,
        const char& ul, 
        const char& ur, 
        const char& l, 
        const char& c, 
        const char& r,
        const char& ll,
        const char& lr)
    {
        state = oldState;
        int flowState = 
            (not0(ul) << 6) +
            (not0(ur) << 5) +
            (not0(l)  << 4) +
            (not0(c)  << 3) +
            (not0(r)  << 2) +
            (not0(ll) << 1) +
            (not0(lr) << 0);
        long tmp = (long)this;
        int rand = tmp;
        rand = simpleRand((rand >> 13) ^ t);
        int tinyRand = rand & 3;
        int bigRand  = rand & 0xff;

        if (oldState == liquid) {
            particles[(int)simParams->transportTable[flowState][tinyRand][0]] = ul;
            particles[(int)simParams->transportTable[flowState][tinyRand][1]] = ur;
            particles[(int)simParams->transportTable[flowState][tinyRand][2]] =  l;
            particles[(int)simParams->transportTable[flowState][tinyRand][3]] =  c;
            particles[(int)simParams->transportTable[flowState][tinyRand][4]] =  r;
            particles[(int)simParams->transportTable[flowState][tinyRand][5]] = ll;
            particles[(int)simParams->transportTable[flowState][tinyRand][6]] = lr;
            return;
        }

        if (oldState == slip) {
            particles[UL] = ur;
            particles[UR] = ul;
            particles[L ] = r;
            particles[C ] = c;
            particles[R ] = l;
            particles[LL] = lr;
            particles[LR] = ll;

            if (particles[UR] == 0) 
                swap(particles[UR], particles[UL]);
            if (particles[LR] == 0) 
                swap(particles[LR], particles[LL]);
            if (particles[R] == 0) 
                swap(particles[R], particles[L]);
                
            return;
        }

        particles[UL] = ul;
        particles[UR] = ur;
        particles[L ] = l;
        particles[C ] = c;
        particles[R ] = r;
        particles[LL] = ll;
        particles[LR] = lr;

        if (state == source) {
            if (bigRand < 8) {
                particles[R] = ((t / simParams->colorSwitchCycles) & 3) + 1;
            }
        }
    } 

    __device__ __host__ inline char& getState() 
    {
        return state;
    }

    __device__ __host__ inline const char& getState() const
    {
        return state;
    }

    __device__ __host__ inline const char& operator[](const int& i) const
    {
        return particles[i];
    }

    inline std::string toString() const
    {
        std::ostringstream buf;
        buf << "(" 
            << stateToString(state) << ", "
            << particleToString(particles[0]) << ", "
            << particleToString(particles[1]) << ", "
            << particleToString(particles[2]) << ", "
            << particleToString(particles[3]) << ", "
            << particleToString(particles[4]) << ", "
            << particleToString(particles[5]) << ", "
            << particleToString(particles[6]) << ")";
        return buf.str();
    }

    static void initTransportTable() 
    {
        Cell::Pattern p;
        p(L, UL);
        p(R, LR);
        std::cout << "L: " << L << "\n"
                  << "R: " << R << "\n"
                  << "UL: " << UL << "\n"
                  << "LR: " << LR << "\n"
                  << "p.isSet(L) = " << p.isSet(L) << "\n";
        std::cout << "\n" << p.toString() << "\n";
        std::cout << "\n" << p.rotate(1).toString() << "\n";
        std::cout << "\n" << p.rotate(2).toString() << "\n";
        std::cout << "\n" << p.rotate(6).toString() << "\n";

        // default: no collision
        for (int i = 0; i < 128; ++i) 
            for (int rand = 0; rand < 4; ++rand)
                fillIn(i, rand, LR, LL, R, C, L, UR, UL);

        // add patterns according to Fig. 7.2:
        {
            // a
            Pattern p[4];
            p[0](L, LR);
            p[0](R, UL);
            p[1](L, UR);
            p[1](R, LL);
            p[2](L, UL);
            p[2](R, LR);
            p[3](L, LL);
            p[3](R, UR);
            addPattern(p, 4);
        }
        {
            // b
            Pattern p[0];
            p[0](UL, UL);
            p[0](R,  R);
            p[0](LL, LL);
            addPattern(p, 1);
        }
        {
            // c
            Pattern p[2];
            p[0](L, UR);
            p[0](C, LR);
            p[1](L, LR);
            p[1](C, UR);
            addPattern(p, 2);
        }
        {
            // d'
            Pattern p[4];
            p[0](UL, C);
            p[0](LL, R);
            p[1](UL, R);
            p[1](LL, C);
            p[2](UL, UR);
            p[2](LL, LR);
            p[3](UL, LR);
            p[3](LL, UR);
            addPattern(p, 4);
        }
        {
            // e
            Pattern p[2];
            p[0](L, UL);
            p[0](C, C);
            p[0](R, LR);
            p[1](L, LL);
            p[1](C, C);
            p[1](R, UR);
            addPattern(p, 2);
        }
        {
            // f
            Pattern p[1];
            p[0](UL, UL);
            p[0](LL, LL);
            p[0](R,  R);
            p[0](C,  C);
            addPattern(p, 1);
        }
        // {
        //     // fixme
        //     Pattern p[2];
        //     p[0](L, UR);
        //     p[0](C, LR);
        //     p[1](L, LR);
        //     p[1](C, UR);
        //     addPattern(p, 2);
        // }

    }

    static void initPalette() 
    {
        simParamsHost.palette[0][0] = 0;
        simParamsHost.palette[0][1] = 0;
        simParamsHost.palette[0][2] = 0;

        simParamsHost.palette[1][0] = 0;
        simParamsHost.palette[1][1] = 0;
        simParamsHost.palette[1][2] = 255;

        simParamsHost.palette[2][0] = 0;
        simParamsHost.palette[2][1] = 255;
        simParamsHost.palette[2][2] = 0;

        simParamsHost.palette[3][0] = 0;
        simParamsHost.palette[3][1] = 0;
        simParamsHost.palette[3][2] = 255;

        simParamsHost.palette[4][0] = 0;
        simParamsHost.palette[4][1] = 255;
        simParamsHost.palette[4][2] = 0;

        for (int i = 5; i < 256; ++i) {
            simParamsHost.palette[i][0] = 255;
            simParamsHost.palette[i][1] = 0;
            simParamsHost.palette[i][2] = 0;
        }
    }

    static void addPattern(const Pattern *p, const int& num)
    {
        for (int angle = 0; angle < 6; ++angle) {
            for (int offset = 0; offset < num; ++offset) {
                Pattern rot = p[offset].rotate(angle);
                addFinalPattern(offset, rot);
                if (num == 2) {
                    addFinalPattern(offset + 2, rot);
                }
                if (num == 1) {
                    addFinalPattern(offset + 1, rot);
                    addFinalPattern(offset + 2, rot);
                    addFinalPattern(offset + 3, rot);
                }
            }
        }
    }

    static void addFinalPattern(const int& offset, const Pattern& p)
    {
        for (int i = 0; i < 7; ++i)
            simParamsHost.transportTable[p.getFlowState()][offset][i] = p.getDest(i);
    } 
    
    char particles[7];
    char state;

private:
    __device__ __host__ inline bool not0(const char& c) const
    {
        return c != 0? 1 : 0;
    }

    inline std::string particleToString(const char& p) const
    {
        if (p == 0)
            return " ";
        if (p == 1)
            return "r";
        if (p == 2)
            return "g";
        if (p == 3)
            return "b";
        if (p == 4)
            return "y";
        return "X";
    }

    inline std::string stateToString(const char& state) const
    {
        switch (state) {
        case liquid:
            return "l";
        case solid:
            return "S";
        case source:
            return "s";
        case drain:
            return "d";
        default:
            return "X";
        }
    }

    static void fillIn(const int& flowState,
                       const int& rand,
                       const char& ul,
                       const char& ur,
                       const char& l,
                       const char& c,
                       const char& r,
                       const char& ll,
                       const char& lr)
    {
        simParamsHost.transportTable[flowState][rand][UL] = ul;
        simParamsHost.transportTable[flowState][rand][UR] = ur;
        simParamsHost.transportTable[flowState][rand][L]  = l;
        simParamsHost.transportTable[flowState][rand][C]  = c;
        simParamsHost.transportTable[flowState][rand][R]  = r;
        simParamsHost.transportTable[flowState][rand][LL] = ll;
        simParamsHost.transportTable[flowState][rand][LR] = lr;
    }
};

#endif
