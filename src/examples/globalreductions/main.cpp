#include <libgeodecomp.h>

using namespace LibGeoDecomp;

class TemperatureRecorder;
class RainMaker;

class BushFireCell
{
public:
    friend void runSimulation();
    friend TemperatureRecorder;
    friend RainMaker;

    enum State {BURNING, GUTTED};

    class API :
        public APITraits::HasFixedCoordsOnlyUpdate
    {};

    inline
    BushFireCell(
        const double humidity = 0,
        const double fuel = 0,
        const double temperature = 0,
        const State state = GUTTED) :
        humidity(humidity),
        fuel(fuel),
        temperature(temperature),
        state(state)
    {}

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, const int /*nanoStep*/)
    {
        temperature = 0.2 * (
            hood[FixedCoord<0, -1>()].temperature +
            hood[FixedCoord<-1, 0>()].temperature +
            hood[FixedCoord< 0, 0>()].temperature +
            hood[FixedCoord< 1, 0>()].temperature +
            hood[FixedCoord<0,  1>()].temperature);

        double evaporation = std::min(1.0, 0.993 + 0.7/ (0.001 + temperature));
        humidity = hood[FixedCoord<>()].humidity * evaporation;

        double lostHumidity = hood[FixedCoord<>()].humidity - humidity;
        temperature -= lostHumidity * 4000;
        if (temperature < 0) {
            temperature = 0;
        }

        fuel = hood[FixedCoord<>()].fuel;
        state = hood[FixedCoord<>()].state;

        if (state == BURNING) {
            temperature += 40.0;
            fuel -= 0.0225;
            // quenching:
            if ((temperature <= 50) || (fuel <= 0) || (humidity >= 0.33)) {
                state = GUTTED;
                fuel = 0;
            }
        } else {
            // ignition:
            if ((temperature > 50) && (fuel > 0) && (humidity < 0.33)) {
                state = BURNING;
            }
        }
    }

private:
    double humidity;
    double fuel;
    double temperature;
    int state;
};

class BushFireInitializer : public SimpleInitializer<BushFireCell>
{
public:
    BushFireInitializer(const Coord<2>& dim, const int maxSteps) :
        SimpleInitializer<BushFireCell>(dim, maxSteps)
    {}


    virtual void grid(GridBase<BushFireCell, 2> *ret)
    {
        Random::seed(4711);
        CoordBox<2> box = ret->boundingBox();
        Grid<double> humidityGrid = createUnwarpedPlasmaField(gridDimensions(), 0.5);
        Grid<double> fuelGrid = createUnwarpedPlasmaField(gridDimensions(), 0.01);

        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            ret->set(*i, BushFireCell(humidityGrid[*i], 1.0 + fuelGrid[*i]));
        }

        CoordBox<2> seatOfFire(Coord<2>(100, 100), Coord<2>(10, 10));

        for (CoordBox<2>::Iterator i = seatOfFire.begin(); i != seatOfFire.end(); ++i) {
            if (box.inBounds(*i)) {
                ret->set(*i, BushFireCell(0, 10.0, 200.0, BushFireCell::BURNING));
            }
        }
    }

private:

    /**
     * accounts for distorions caused in createPlasmaField() by aspect
     * ratio and amplitude/coarseness.
     */
    Grid<double> createUnwarpedPlasmaField(const Coord<2>& dim, const double coarseness)
    {
        Coord<2> enclosingSquare = Coord<2>::diagonal(std::max(dim.x(), dim.y()));
        double amplitude = 10.0;
        double minValue = std::numeric_limits<double>::max();
        double maxValue = std::numeric_limits<double>::min();

        Grid<double> grid = createPlasmaField(enclosingSquare, amplitude, coarseness);

        Grid<double> ret(dim);

        for (int y = 0; y < dim.y(); ++y) {
            for (int x = 0; x < dim.x(); ++x) {
                Coord<2> c(x, y);

                if (grid[c] < minValue) {
                    minValue = grid[c];
                }

                if (grid[c] > maxValue) {
                    maxValue = grid[c];
                }
            }
        }

        double scale = 1.0 / (maxValue - minValue);

        for (int y = 0; y < dim.y(); ++y) {
            for (int x = 0; x < dim.x(); ++x) {
                Coord<2> c(x, y);

                ret[c] = (grid[c] - minValue) * scale;
            }
        }

        return ret;
    }


    Grid<double> createPlasmaField(const Coord<2>& dim, const double amplitude, const double coarseness)
    {
        Grid<double> ret(dim);
        ret[Coord<2>(0,           0          )] = (Random::gen_d(1.0) - 0.5) * amplitude;
        ret[Coord<2>(dim.x() - 1, 0          )] = (Random::gen_d(1.0) - 0.5) * amplitude;
        ret[Coord<2>(0,           dim.y() - 1)] = (Random::gen_d(1.0) - 0.5) * amplitude;
        ret[Coord<2>(dim.x() - 1, dim.y() - 1)] = (Random::gen_d(1.0) - 0.5) * amplitude;

        CoordBox<2> box(Coord<2>(), dim - Coord<2>::diagonal(1));

        fillRect(&ret, box, amplitude, coarseness);

        return ret;
    }

    void fillRect(Grid<double> *ret, const CoordBox<2>& box, const double amplitude, const double coarseness)
    {
        if ((box.dimensions.x() <= 1) && (box.dimensions.y() <= 1)) {
            return;
        }

        int leftX = box.origin.x();
        int rightX = box.origin.x() + box.dimensions.x();

        int upperY = box.origin.y();
        int lowerY = box.origin.y() + box.dimensions.y();

        int midX = box.origin.x() + box.dimensions.x() / 2;
        int midY = box.origin.y() + box.dimensions.y() / 2;

        Coord<2> upperLeft( leftX,  upperY);
        Coord<2> upperRight(rightX, upperY);
        Coord<2> lowerLeft( leftX,  lowerY);
        Coord<2> lowerRight(rightX, lowerY);

        Coord<2> upperMiddle(midX,   upperY);
        Coord<2> lowerMiddle(midX,   lowerY);
        Coord<2> leftMiddle( leftX,  midY);
        Coord<2> rightMiddle(rightX, midY);
        Coord<2> center(midX, midY);

        if (box.dimensions.x() > 1) {
            ret->set(upperMiddle, 0.5 * (ret->get(upperLeft) + ret->get(upperRight)));
            ret->set(lowerMiddle, 0.5 * (ret->get(lowerLeft) + ret->get(lowerRight)));
        }

        if (box.dimensions.y() > 1) {
            ret->set(leftMiddle,  0.5 * (ret->get(upperLeft)  + ret->get(lowerLeft)));
            ret->set(rightMiddle, 0.5 * (ret->get(upperRight) + ret->get(lowerRight)));
        }

        if ((box.dimensions.x() > 1) && (box.dimensions.y() > 1)) {
            ret->set(
                center,
                0.25 * (ret->get(upperLeft) +
                        ret->get(upperRight) +
                        ret->get(lowerLeft) +
                        ret->get(lowerRight)) +
                (Random::gen_d(1.0) - 0.5) * (amplitude + coarseness));
        }

        Coord<2> halfDim = box.dimensions / 2;
        Coord<2> remainder = box.dimensions - halfDim;
        Coord<2> lowerLeftQuarter(halfDim.x(), remainder.y());
        Coord<2> lowerRightQuarter(remainder.x(), halfDim.y());
        double newAmplitude = amplitude * 0.5;

        fillRect(ret, CoordBox<2>(upperLeft,   halfDim),           newAmplitude, coarseness);
        fillRect(ret, CoordBox<2>(leftMiddle,  lowerLeftQuarter),  newAmplitude, coarseness);
        fillRect(ret, CoordBox<2>(upperMiddle, lowerRightQuarter), newAmplitude, coarseness);
        fillRect(ret, CoordBox<2>(center,      remainder),         newAmplitude, coarseness);
    }
};

class TemperatureRecorder : public Clonable<Writer<BushFireCell>, TemperatureRecorder>
{
public:
    typedef typename Writer<BushFireCell>::GridType GridType;

    TemperatureRecorder(const unsigned outputPeriod) :
        Clonable<Writer<BushFireCell>, TemperatureRecorder>("", outputPeriod),
        avrgTemperature(0)
    {}

    void stepFinished(const GridType& grid, unsigned step, WriterEvent event)
    {
        avrgTemperature = 0;

        CoordBox<2> box = grid.boundingBox();
        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            avrgTemperature += grid.get(*i).temperature;
        }

        avrgTemperature /= box.dimensions.prod();
        std::cout << "averageTemperature(" << step << ") = " << avrgTemperature << "\n";
    }

    double averageTemperature() const
    {
        return avrgTemperature;
    }

private:
    double avrgTemperature;
};

class RainMaker : public Steerer<BushFireCell>
{
public:
    using Steerer<BushFireCell>::CoordType;
    using Steerer<BushFireCell>::GridType;
    using Steerer<BushFireCell>::Topology;

    RainMaker(const unsigned ioPeriod, TemperatureRecorder *trigger) :
        Steerer<BushFireCell>(ioPeriod),
        waterAvailable(true),
        trigger(trigger)
    {}

    void nextStep(
        GridType *grid,
        const Region<Topology::DIM>& validRegion,
        const CoordType& globalDimensions,
        unsigned step,
        SteererEvent event,
        std::size_t rank,
        bool lastCall,
        SteererFeedback *feedback)
    {
        if (waterAvailable && (trigger->averageTemperature() > 250)) {
            std::cout << "WARNING---------------------------------------------------\n"
                      << "WARNING: initiating rain at time step " << step << "\n"
                      << "WARNING---------------------------------------------------\n";

            for (Region<Topology::DIM>::Iterator i = validRegion.begin();
                 i != validRegion.end();
                 ++i) {
                BushFireCell cell = grid->get(*i);
                cell.humidity += 0.25;
                grid->set(*i, cell);
            }

            if (lastCall) {
                waterAvailable = 0;
            }
        }
    }

private:
    bool waterAvailable;
    TemperatureRecorder *trigger;
};

void runSimulation()
{
    Coord<2> dim(1000, 500);
    int maxSteps = 5000;
    int outputPeriod = 5;

    SerialSimulator<BushFireCell> sim(new BushFireInitializer(dim, maxSteps));

    sim.addWriter(new SerialBOVWriter<BushFireCell>(&BushFireCell::humidity,    "humidity",
                      outputPeriod));
    sim.addWriter(new SerialBOVWriter<BushFireCell>(&BushFireCell::fuel,        "fuel",
                      outputPeriod));
    sim.addWriter(new SerialBOVWriter<BushFireCell>(&BushFireCell::temperature, "temperature",
                      outputPeriod));
    sim.addWriter(new SerialBOVWriter<BushFireCell>(&BushFireCell::state,       "state",
                      outputPeriod));

    sim.addWriter(new TracingWriter<BushFireCell>(500, maxSteps));

    TemperatureRecorder *temperatureRecorder = new TemperatureRecorder(100);
    sim.addWriter(temperatureRecorder);
    sim.addSteerer(new RainMaker(100, temperatureRecorder));

    sim.run();
}

int main(int argc, char **argv)
{
    runSimulation();
    return 0;
}
