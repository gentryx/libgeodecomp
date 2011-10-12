#include <iostream>
#include <sstream>

class Cell
{
public:
    // defines for each of the 2^7 flow states which particle moves to
    // which position. stores four variants since the FHP-II model
    // sometimes requires a probabilistic selection. Don't pad to 8
    // bytes to reduce bank conflicts on Nvidia GPUs.
    static char transportTable[128][4][7];

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
        bounce,
        source,
        drain
    };

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

    inline void update(
        const int& randSeed,
        const char& oldState,
        const char& ul, 
        const char& ur, 
        const char& l, 
        const char& c, 
        const char& r,
        const char& ll,
        const char& lr)
    {
        int flowState = 
            (not0(ul) << 6) +
            (not0(ur) << 5) +
            (not0(l)  << 4) +
            (not0(c)  << 3) +
            (not0(r)  << 2) +
            (not0(ll) << 1) +
            (not0(lr) << 0);

        int rand = ((randSeed ^ flowState) >> 3) & 3;
        if (flowState == 20) 
            std::cout << "yuk: " << flowState << " and " << rand << "\n";

        particles[(int)transportTable[flowState][rand][0]] = ul;
        particles[(int)transportTable[flowState][rand][1]] = ur;
        particles[(int)transportTable[flowState][rand][2]] =  l;
        particles[(int)transportTable[flowState][rand][3]] =  c;
        particles[(int)transportTable[flowState][rand][4]] =  r;
        particles[(int)transportTable[flowState][rand][5]] = ll;
        particles[(int)transportTable[flowState][rand][6]] = lr;
        state = oldState;
    } 

    inline const char& getState() const
    {
        return state;
    }

    inline const char& operator[](const int& i) const
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
            Pattern p[2];
            p[0](L, LR);
            p[0](R, UL);
            p[1](L, UR);
            p[1](R, LL);
            addPattern(p, 2);
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
            // fixme
            // Pattern p[2];
            // p[0](L, UR);
            // p[0](C, LR);
            // p[0](L, LR);
            // p[0](C, UR);
            // addPattern(p, 2);
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
            transportTable[p.getFlowState()][offset][i] = p.getDest(i);
    } 
    
private:
    char particles[7];
    char state;

    inline bool not0(const char& c) const
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
        case bounce:
            return "b";
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
        transportTable[flowState][rand][UL] = ul;
        transportTable[flowState][rand][UR] = ur;
        transportTable[flowState][rand][L]  = l;
        transportTable[flowState][rand][C]  = c;
        transportTable[flowState][rand][R]  = r;
        transportTable[flowState][rand][LL] = ll;
        transportTable[flowState][rand][LR] = lr;
    }
};


int simpleRand(int i) {
    return i * 69069 + 1327217885;
}

char Cell::transportTable[128][4][7];

void testModel()
{
    int width = 7;
    int height = 6;
    Cell gridA[height][width][2];
    Cell gridB[height][width][2];
    // gridA[1][1][0] = Cell(Cell::liquid, Cell::R);
    // gridA[1][1][0] = Cell(Cell::liquid, Cell::LR);
    // gridA[1][1][1] = Cell(Cell::liquid, Cell::C);
    // gridA[1][5][0] = Cell(Cell::liquid, Cell::LL);
    // gridA[4][1][0] = Cell(Cell::liquid, Cell::UR);
    // gridA[4][5][1] = Cell(Cell::liquid, Cell::UL);
    // gridA[4][5][1] = Cell(Cell::liquid, Cell::L);

    gridA[3][1][0] = Cell(Cell::liquid, Cell::R);
    gridA[3][3][0] = Cell(Cell::liquid, Cell::L);


    for (int t = 0; t < 5; ++t) {
        std::cout << "t: " << t << "\n";

        for (int y = 0; y < height; ++y) {
            for (int ly = 0; ly < 2; ++ly) {
                std::cout << "(" << y << ", " << ly << ") ";
                for (int x = 0; x < width; ++x) {
                    std::cout << gridA[y][x][ly].toString();
                }
                std::cout << "\n";
            }
        }

        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                gridB[y][x][0].update(
                    simpleRand(x + y * 2 * width + 0 * width + t),
                    gridA[y + 0][x + 0][0].getState(),
                    gridA[y - 1][x + 0][1][Cell::LR],
                    gridA[y - 1][x + 1][1][Cell::LL],
                    gridA[y + 0][x - 1][0][Cell::R ],
                    gridA[y + 0][x + 0][0][Cell::C ],
                    gridA[y + 0][x + 1][0][Cell::L ],
                    gridA[y + 0][x + 0][1][Cell::UR],
                    gridA[y + 0][x + 1][1][Cell::UL]);

                gridB[y][x][1].update(
                    simpleRand(x + y * 2 * width + 1 * width + t),
                    gridA[y + 0][x + 0][1].getState(),
                    gridA[y + 0][x - 1][0][Cell::LR],
                    gridA[y + 0][x + 0][0][Cell::LL],
                    gridA[y + 0][x - 1][1][Cell::R ],
                    gridA[y + 0][x + 0][1][Cell::C ],
                    gridA[y + 0][x + 1][1][Cell::L ],
                    gridA[y + 1][x - 1][0][Cell::UR],
                    gridA[y + 1][x + 0][0][Cell::UL]);
            }
        }


        for (int y = 0; y < height; ++y) {
            for (int ly = 0; ly < 2; ++ly) {
                for (int x = 0; x < width; ++x) {
                    gridA[y][x][ly] = gridB[y][x][ly];

                }
            }
        }

        std::cout << "\n";
    }
}

int main(int argc, char **argv)
{
    std::cout << "ok\n";
    Cell::initTransportTable();

    // testModel();

    

    return 0;
}
