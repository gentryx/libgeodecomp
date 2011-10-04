#include <iostream>
#include <sstream>

class Cell
{
public:
    static char transportTable[128][8];

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

    Cell(int k=-1) :
        state(liquid)
    {
        for (int i = 0; i < 7; ++i)
            particles[i] = 0;
        if (k >= 0)
            particles[k] = 1;
    }

    void update(const State& state,
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
        
        particles[(int)transportTable[flowState][0]] = ul;
        particles[(int)transportTable[flowState][1]] = ur;
        particles[(int)transportTable[flowState][2]] =  l;
        particles[(int)transportTable[flowState][3]] =  c;
        particles[(int)transportTable[flowState][4]] =  r;
        particles[(int)transportTable[flowState][5]] = ll;
        particles[(int)transportTable[flowState][6]] = lr;
    } 

    const State& getState() const
    {
        return state;
    }

    const char& operator[](const int& i) const
    {
        return particles[i];
    }

    std::string toString() const
    {
        std::ostringstream buf;
        buf << "(" 
            << particleToString(particles[0]) << ", "
            << particleToString(particles[1]) << ", "
            << particleToString(particles[2]) << ", "
            << particleToString(particles[3]) << ", "
            << particleToString(particles[4]) << ", "
            << particleToString(particles[5]) << ", "
            << particleToString(particles[6]) << ")";
        return buf.str();
    }

    
private:
    char particles[7];
    State state;

    bool not0(const char& c)
    {
        return c != 0? 1 : 0;
    }

    std::string particleToString(const char& p) const
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
};

char Cell::transportTable[128][8];

int main(int argc, char **argv)
{
    std::cout << "ok\n";
    for (int i = 0; i < 128; ++i) {
        Cell::transportTable[i][Cell::UL] = Cell::LR;
        Cell::transportTable[i][Cell::UR] = Cell::LL;
        Cell::transportTable[i][Cell::L ] = Cell::R;
        Cell::transportTable[i][Cell::C ] = Cell::C;
        Cell::transportTable[i][Cell::R ] = Cell::L;
        Cell::transportTable[i][Cell::LL] = Cell::UR;
        Cell::transportTable[i][Cell::LR] = Cell::UL;
    }
    
    int width = 7;
    int height = 6;
    Cell gridA[height][width][2];
    Cell gridB[height][width][2];
    gridA[1][1][0] = Cell(Cell::R);

    for (int t = 0; t < 5; ++t) {
        std::cout << "t: " << t << "\n";

        for (int y = 0; y < height; ++y) {
            for (int ly = 0; ly < 2; ++ly) {
                for (int x = 0; x < width; ++x) {
                    std::cout << gridA[y][x][ly].toString();
                }
                std::cout << "\n";
            }
        }

        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                gridB[y][x][0].update(
                    gridA[y + 0][x + 0][0].getState(),
                    gridA[y - 1][x + 0][1][Cell::LR],
                    gridA[y - 1][x + 1][1][Cell::LL],
                    gridA[y + 0][x - 1][0][Cell::R ],
                    gridA[y + 0][x + 0][0][Cell::C ],
                    gridA[y + 0][x + 1][0][Cell::L ],
                    gridA[y + 0][x + 0][1][Cell::UR],
                    gridA[y + 0][x + 1][1][Cell::UL]);

                gridB[y][x][1].update(
                    gridA[y + 0][x + 0][1].getState(),
                    gridA[y + 0][x - 1][0][Cell::LR],
                    gridA[y + 0][x + 0][0][Cell::LL],
                    gridA[y + 0][x - 1][1][Cell::R ],
                    gridA[y + 0][x + 0][1][Cell::C ],
                    gridA[y + 0][x - 1][1][Cell::L ],
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

    return 0;
}
