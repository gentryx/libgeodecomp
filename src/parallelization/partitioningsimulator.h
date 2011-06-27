#include <libgeodecomp/config.h>
#ifdef LIBGEODECOMP_FEATURE_MPI
#ifndef _libgeodecomp_parallelization_partitioningsimulator_h_
#define _libgeodecomp_parallelization_partitioningsimulator_h_

#include <set>
#include <map>
#include <iostream>
#include <math.h>
#include <unistd.h>
#include <libgeodecomp/misc/grid.h>
#include <libgeodecomp/misc/coordbox.h>
#include <libgeodecomp/misc/displacedgrid.h>
#include <libgeodecomp/misc/stringops.h>
#include <libgeodecomp/misc/supermap.h>
#include <libgeodecomp/misc/supervector.h>
#include <libgeodecomp/mpilayer/mpilayer.h>
#include <libgeodecomp/parallelization/monolithicsimulator.h>
#include <libgeodecomp/parallelization/chronometer.h>
#include <libgeodecomp/parallelization/partitioningsimulator/commjob.h>
#include <libgeodecomp/parallelization/partitioningsimulator/coordset.h>
#include <libgeodecomp/parallelization/partitioningsimulator/partition.h>
#include <libgeodecomp/parallelization/partitioningsimulator/partitionstriping.h>
#include <libgeodecomp/parallelization/partitioningsimulator/partitionrecursivebisection.h>
#include <libgeodecomp/parallelization/partitioningsimulator/modelsplitter.h>
#include <libgeodecomp/parallelization/partitioningsimulator/staticloadmodel.h>
#include <libgeodecomp/parallelization/partitioningsimulator/homogeneousloadmodel.h>
#include <libgeodecomp/parallelization/partitioningsimulator/statebasedloadmodel.h>
#include <libgeodecomp/parallelization/partitioningsimulator/communicationmodel.h>

namespace LibGeoDecomp {

/**
 * This is the main application class, it performs the iteration of
 * simulation steps. 
 */
template<typename CELL_TYPE>
class PartitioningSimulator : public MonolithicSimulator<CELL_TYPE>
{
    friend class PartitioningSimulatorTest;
    friend class PartitioningSimulatorParallel4Test;

public:

    PartitioningSimulator(
        Initializer<CELL_TYPE> *_initializer,
        std::string partitionType = "recursive_bisection",
        std::string loadModelType = "state_based",
        std::string clusterConfig = "", 
        const unsigned& loadBalancingEveryN = 16,
        const bool& verbose = false,
        const MPI::Datatype& cellMPIDatatype = Typemaps::lookup<CELL_TYPE>(),
        std::ostream& reportStream = std::cout): 
        MonolithicSimulator<CELL_TYPE>(_initializer),
        _rank(_mpilayer.rank()),
        _width(this->initializer->gridDimensions().x()),
        _height(this->initializer->gridDimensions().y()),
        _rimWidth(1),
        _loadBalancingEveryN(loadBalancingEveryN),
        _verbose(verbose),
        _master(0),
        _reportStream(reportStream),
        _loadModel(0),
        _communicationModel(0),
        _clusterTable(0),
        _cellMPIDatatype(cellMPIDatatype),
        _partitionType(partitionType)
    {
        initClusterTable(clusterConfig);
        _communicationModel = new CommunicationModel(*_clusterTable);
        initLoadModel(loadModelType);

        _partition = createNewPartition();
        _partition->copyFromBroadcast(_mpilayer, _master); 

        _loadModel->repartition(*_partition);

        initializeRim(_partition);

        CoordBox<2> rect = _simulationRectangle;
        _newGrid = new DisplacedGrid<CELL_TYPE>(rect);
        _curGrid = new DisplacedGrid<CELL_TYPE>(rect);
        this->initializer->grid(_newGrid);
        this->initializer->grid(_curGrid);

        _recvOuterRimJobs = createJobsFromCoords(
                recvOuterRimMap(_partition), *_newGrid);
        _sendInnerRimJobs = createJobsFromCoords(
                sendInnerRimMap(_partition), *_newGrid);
    }


    ~PartitioningSimulator()
    {
        delete _curGrid;
        delete _newGrid;
        delete _loadModel;
        delete _communicationModel;
        delete _clusterTable;        
        delete _partition;
    }


    void step()
    {
        balanceLoad();
        for (unsigned i = 0; i < CELL_TYPE::nanoSteps(); i++) 
            nanoStep(i);

        this->stepNum++;    
        // call back all registered Writers
        for(unsigned i = 0; i < this->writers.size(); i++) 
            this->writers[i]->stepFinished();
    }


    inline void run()
    {
        for (unsigned i = 0; i < this->writers.size(); i++) 
            this->writers[i]->initialized();

        for (unsigned i = 0; i < this->initializer->maxSteps(); i++) 
            step();

        for (unsigned i = 0; i < this->writers.size(); i++) 
            this->writers[i]->allDone();        

        if (_verbose) summary();
    }


    inline Grid<CELL_TYPE> *getGrid()
    {
        return _curGrid->vanillaGrid();
    }


    inline DisplacedGrid<CELL_TYPE> *getDisplacedGrid()
    {
        return _curGrid;
    }


    inline const CoordBox<2>& partitionRectangle() const
    {
        return _partitionRectangle;
    }

    
    inline const LoadModel *getLoadModel() const
    {
        return _loadModel;
    }

    inline const Partition *getPartition() const
    {
        return _partition;
    }

private:
    typedef SuperMap<unsigned, Coord<2>::Vector> UCVMap;
    typedef SuperVector<MPILayer::MPIRegionPointer> RegionVector;
    typedef SuperVector<UVec> StateCounts;

    MPILayer _mpilayer;
    unsigned const _rank;

    DisplacedGrid<CELL_TYPE> *_curGrid;
    DisplacedGrid<CELL_TYPE> *_newGrid;
    unsigned _width;
    unsigned _height;
    unsigned _rimWidth;

    /**
     * the Partition for the whole SimulationGrid
     */
    Partition* _partition;

    /**
     * the area of responsibility for this node in absolute coordinates
     */
    CoordBox<2> _partitionRectangle;

    /**
     * the _partitionRectangle, increased by _rimWidth, absolute coordinates
     */
    CoordBox<2> _simulationRectangle;

    /**
     * core in absolute coordinates
     */
    CoordBox<2> _coreRectangle;

    /**
     * innerRim in absolute coordinates
     */
    CoordSet _innerRim;

    CommJobs _sendInnerRimJobs;
    CommJobs _recvOuterRimJobs;

    RegionVector _registeredRegions;

    unsigned _loadBalancingEveryN;
    Chronometer _updateTimer;
    Chronometer _balanceTimer;
    Chronometer _repartitionTimer;
    // these timers don't include repartitioning
    Chronometer _totalUpdateTimer;
    Chronometer _totalSendTimer;
    Chronometer _totalRecvTimer;
    Chronometer _totalWaitTimer;

    bool _verbose;

    /**
     * the node with _rank _master does the talking
     */
    unsigned _master;
    std::ostream& _reportStream;

    LoadModel* _loadModel;
    CommunicationModel* _communicationModel; 
    ClusterTable* _clusterTable;
    MPI::Datatype _cellMPIDatatype;

    std::string _partitionType;

    /** 
     * initializes _partitionRectangle, _simulationRectangle, _coreRectangle, and _innerRim. 
     */
    void initializeRim(Partition* partition)
    {
        _partitionRectangle = partition->coordsForNode(_rank); 
        _innerRim = CoordSet();
        CoordBox<2> pRect = _partitionRectangle;

        //we don't need to bother with rims unless we actually have work to do
        if (pRect.size() == 0) {
            _simulationRectangle = pRect;
            _coreRectangle = pRect;
        } else {
            initializeRealRim(partition);
        }
    }


    void initializeRealRim(Partition* partition)
    {
        // fixme shortcut because kobo was too lazy to type _partitionRectangle over and over again ;-)
        // or maybe he just thought it was too ugly to repeatedly read such a long specifier.
        // maybe we should then rename the member to pRect.
        CoordBox<2> pRect = _partitionRectangle;

        Coord<2> origin = pRect.origin; 
        Coord<2> originOpposite = pRect.originOpposite();

        bool leftRim = partition->inBounds(origin - Coord<2>(_rimWidth, 0));
        bool rightRim = partition->inBounds(originOpposite + Coord<2>(_rimWidth, 0));
        bool upperRim = partition->inBounds(origin - Coord<2>(0, _rimWidth));
        bool lowerRim = partition->inBounds(originOpposite + Coord<2>(0, _rimWidth));

        unsigned leftInc = leftRim ? _rimWidth : 0;
        unsigned rightInc = rightRim ? _rimWidth : 0;
        unsigned upperInc = upperRim ? _rimWidth : 0;
        unsigned lowerInc = lowerRim ? _rimWidth : 0;
        _simulationRectangle = CoordBox<2>(
            pRect.origin - Coord<2>(leftInc, upperInc),
            Coord<2>(
                pRect.dimensions.x() + leftInc + rightInc,
                pRect.dimensions.y() + upperInc + lowerInc));


        unsigned coreWidth = pRect.dimensions.x();
        unsigned coreHeight = pRect.dimensions.y();
        Coord<2> coreOrigin = pRect.origin; 

        if (leftRim) {
            coreOrigin += Coord<2>(_rimWidth, 0);
            coreWidth -= _rimWidth;
            CoordBox<2> left(pRect.origin, 
                             Coord<2>(_rimWidth, pRect.dimensions.y()));
            _innerRim.insert(left);
        }
        if (rightRim) {
            coreWidth -= _rimWidth;
            Coord<2> rightRimCorner = pRect.origin + 
                Coord<2>(pRect.dimensions.x() - _rimWidth, 0);
            CoordBox<2> right(rightRimCorner, 
                              Coord<2>(_rimWidth, pRect.dimensions.y()));
            _innerRim.insert(right);
        }
        coreWidth = std::max((int)coreWidth, 0);

        if (upperRim) {
            coreOrigin += Coord<2>(0, _rimWidth);
            coreHeight -= _rimWidth;
            CoordBox<2> upper(pRect.origin,
                              Coord<2>(pRect.dimensions.x(), _rimWidth));
            _innerRim.insert(upper);
        }
        if (lowerRim) {
            coreHeight -= _rimWidth;
            Coord<2> lowerRimCorner = pRect.origin + 
                Coord<2>(0, pRect.dimensions.y() - _rimWidth);
            CoordBox<2> lower(lowerRimCorner, 
                              Coord<2>(pRect.dimensions.x(), _rimWidth));
            _innerRim.insert(lower);
        }
        coreHeight = std::max((int)coreHeight, 0);
        _coreRectangle = CoordBox<2>(coreOrigin, 
                                     Coord<2>(coreWidth, coreHeight)); 
    }

    std::set<unsigned> findRecipients(
            const Coord<2>& absCoord, const Partition* partition) const
    {
        std::set<unsigned> result;
        Coord<2>::Vector neighbors = absCoord.getNeighbors8();
        for (Coord<2>::Vector::iterator i = neighbors.begin(); 
             i != neighbors.end(); i++) {
            Coord<2> absNeighborCoord = *i;
            if (partition->inBounds(absNeighborCoord)) {
                unsigned dest = partition->nodeForCoord(absNeighborCoord);
                if (dest != _rank)
                    result.insert(dest);
            }
        }
        return result;
    }


    /**
     * return true if the cell at this coord must be aquired from another node
     */
    bool requiredCoord(const Coord<2>& absCoord) const
    {
        Coord<2>::Vector neighbors = absCoord.getNeighbors8();
        for (Coord<2>::Vector::iterator i = neighbors.begin(); 
             i != neighbors.end(); i++) 
            if (_partitionRectangle.inBounds(*i)) return true;
        return false;
    }


    /**
     * Creates the CoordMaps for sendJobs and recvJobs as an intermediate step
     */
    UCVMap sendInnerRimMap(const Partition* partition) const
    {
        SuperMap<unsigned, Coord<2>::Vector>  sendCoords;
        std::set<unsigned> recipients;
        for (CoordBoxSequence<2> s = _partitionRectangle.sequence();
                s.hasNext();) {
                Coord<2> absCoord = s.next();
                if (!_coreRectangle.inBounds(absCoord)) { // belongs to innerRim
                    recipients = findRecipients(absCoord, partition);
                    for (std::set<unsigned>::iterator dest = recipients.begin();
                            dest != recipients.end(); dest ++) 
                        sendCoords[*dest].push_back(absCoord); 

                }
        }
        return sendCoords;
    }

    
    UCVMap recvOuterRimMap(const Partition* partition) const
    {
        SuperMap<unsigned, Coord<2>::Vector> recvCoords;
        for (CoordBoxSequence<2> s = _simulationRectangle.sequence();
                s.hasNext();) {
                Coord<2> absCoord = s.next();
                if ((!_partitionRectangle.inBounds(absCoord)) 
                        && requiredCoord(absCoord)) { // belongs to outerRim
                    unsigned src = partition->nodeForCoord(absCoord);
                    recvCoords[src].push_back(absCoord);
                }
        }
        return recvCoords;
    }


    MPILayer::MPIRegionPointer createRegion(
            const Coord<2>::Vector& coords, 
            DisplacedGrid<CELL_TYPE>& grid, 
            const Coord<2>& base)
    {
        SuperVector<CELL_TYPE*> disps;
        SuperVector<unsigned> lengths;

        Coord<2> prev = coords.front();
        disps.push_back(&(grid[prev]));
        lengths.push_back(1);

        for(Coord<2>::Vector::const_iterator i = ++coords.begin(); 
                i != coords.end(); i++) {
            if ((i->y() == prev.y()) && (i->x() == prev.x() + 1)) {
                lengths.back() += 1;
            } else {
                disps.push_back(&(grid[*i]));
                lengths.push_back(1);
            }
            prev = *i;
        }
        MPILayer::MPIRegionPointer newRegion = 
            _mpilayer.registerRegion(&(grid[base]), disps, lengths, _cellMPIDatatype);
        _registeredRegions.push_back(newRegion);
        return newRegion;
    }


    CommJobs createJobsFromCoords(const UCVMap& map, DisplacedGrid<CELL_TYPE>& grid)
    {
        CommJobs result;
        Coord<2> base = grid.boundingBox().origin;
        for(UCVMap::const_iterator i = map.begin(); i != map.end(); i++) {
            MPILayer::MPIRegionPointer r = createRegion(i->second, grid, base);
            CommJob c(base, r, i->first);
            result.push_back(c);
        }
        return result;
    }


    void nanoStep(const unsigned& nanoStep)
    {
        _totalRecvTimer.tic();
        receive(_newGrid, _recvOuterRimJobs); 
        _totalRecvTimer.toc();

        update(_innerRim.sequence(), nanoStep);

        _totalSendTimer.tic();
        send(_newGrid, _sendInnerRimJobs);
        _totalSendTimer.toc();

        update(_coreRectangle.sequence(), nanoStep);

        _totalWaitTimer.tic();
        finishCommunication();
        _totalWaitTimer.toc();

        std::swap(_curGrid, _newGrid);
    }


    void receive(DisplacedGrid<CELL_TYPE>* target, const CommJobs& jobs)
    {
        for(CommJobs::const_iterator i = jobs.begin(); i != jobs.end(); i++) 
            _mpilayer.recvRegion(&((*target)[i->baseCoord]), i->region, i->partner);
    }


    void send(DisplacedGrid<CELL_TYPE>* source, const CommJobs& jobs)
    {
        for(CommJobs::const_iterator i = jobs.begin(); i != jobs.end(); i++) 
            _mpilayer.sendRegion(&((*source)[i->baseCoord]), i->region, i->partner);
    }


    void finishCommunication()
    {
        _mpilayer.waitAll(); 
    }


    /**
     * when working with rimWidth > 1, updateCore will corrupt the outerRim
     * Therefore we must store the outerRim someplace safe and merge the data
     * after all calculations have finished
     */
    // void mergeData();

    /**
     * node 0 : assemble the complete Grid from all nodes
     * any other node: return an unspecified Grid;
     * this method is mainly for testing.
     */
    Grid<CELL_TYPE> gatherWholeGrid()
    {
        Grid<CELL_TYPE> result(Coord<2>(_width, _height));
        unsigned master = 0;

        // everyone sends own part
        DisplacedGrid<CELL_TYPE> *grid = getDisplacedGrid();
        CoordBox<2> rect = _partitionRectangle;
        rect.origin -= grid->boundingBox().origin;
           
        _mpilayer.sendGridBox(
            grid->vanillaGrid(), rect, master, _cellMPIDatatype);

        // master receives
        if (_rank == master) {
            for (unsigned i = 0; i < _partition->getNumPartitions(); i++) {
                CoordBox<2> rect = _partition->coordsForNode(i);
                _mpilayer.recvGridBox(&result, rect, i, _cellMPIDatatype);
            }
        } 
        _mpilayer.waitAll();
        result.getEdgeCell() = grid->getEdgeCell();
        return result;
    }


    std::string commPos() const
    {
        return "rank " + StringConv::itoa(_rank) + " of " + 
            StringConv::itoa(_mpilayer.size());
    }


    template<typename Sequence>
    void update(Sequence absCoords, const unsigned& nanoStep)
    {
        _totalUpdateTimer.tic();
        _updateTimer.tic();
        while (absCoords.hasNext()) {            
            Coord<2> coord = absCoords.next();
            CoordMap<CELL_TYPE> neighborhood = _curGrid->getNeighborhood(coord);
            (*_newGrid)[coord].update(neighborhood, nanoStep);
        }
        _totalUpdateTimer.toc();
        _updateTimer.toc();
    }


    void initLoadModel(const std::string& type)
    {
        if (type == "state_based") {
            _loadModel = new StateBasedLoadModel(
                    &_mpilayer, 
                    _master, 
                    CoordBox<2>(Coord<2>(0, 0), Coord<2>(_width, _height)),
                    CELL_TYPE::numStates());
        } else if (type == "homogeneous") {
            _loadModel = new HomogeneousLoadModel(&_mpilayer, _master);
        } else if (type == "static") {
            _loadModel = new StaticLoadModel(&_mpilayer, _master);
        } else {        
            throw std::invalid_argument("Unknown loadModel: " + type);
        }
    }


    void initClusterTable(const std::string& clusterConfig)
    {
        _clusterTable = new ClusterTable(_mpilayer.size());
        if (clusterConfig != "") {
            // fixme:
            std::cout << "fixme: ohoh";            
            // _clusterTable = new ClusterTable(_mpilayer, clusterConfig);
        }
    }


    void registerCellsWithLoadModel()
    {
        for (CoordBoxSequence<2> s = _partitionRectangle.sequence();
                s.hasNext();) {
            Coord<2> coord(s.next());
            _loadModel->registerCellState(coord, ((*_curGrid)[coord]).approximateState());
        }
    }


    void balanceLoad()
    {
        // don't balance on the first step since we don't have any data then
        if (this->stepNum % _loadBalancingEveryN != 0) return;
        if (this->stepNum == 0) {
            registerCellsWithLoadModel();
            return;
        }

        _balanceTimer.tic();

        long long cycleLength, workLength;
        _updateTimer.nextCycle(&cycleLength, &workLength);
        _loadModel->sync(*_partition, workLength);
        DVec times = _mpilayer.gather((double)workLength, _master);
        repartitionIfNecessary();
        registerCellsWithLoadModel();

        if (_rank == _master && _verbose) {
            unsigned lastBalanceStep = this->stepNum - _loadBalancingEveryN;
            double predictedRunningTime = 
                _loadModel->predictRunningTime(*_partition);

            Nodes nodes = _partition->getNodes();
            UVec sizes;
            for (Nodes::iterator i = nodes.begin(); i != nodes.end(); i++) 
                sizes.push_back(_partition->coordsForNode(*i).size());

            std::cout
                << "Balancing information for step " << lastBalanceStep 
                << " to " << this->stepNum << ":\n"
                //<< "observed running time: " << (double)workLength << "\n"
                << "observed running time: " << times << "\n" 
                << _loadModel->report()
                << _communicationModel->report()
                << "partition sizes " << sizes << "\n"
                << "predicted running time for next cycle: " 
                << predictedRunningTime << "\n";
        }

        _balanceTimer.toc();
    }


    /**
     * uses _loadModel to create a new Partition.
     */
    Partition* createNewPartition()
    {
        Splitter::SplitDirection direction = Splitter::LONGEST;

        if (_partitionType == "recursive_bisection") {
            // default
        } else if (_partitionType == "striping") {
            direction = Splitter::VERTICAL;
        } else {        
            throw std::invalid_argument("Unknown partition: " + _partitionType);
        }

        if (_rank == _master) { // the real thing
            return new PartitionRecursiveBisection(
                CoordBox<2>(Coord<2>(0, 0), Coord<2>(_width, _height)), 
                Nodes::firstNum(_mpilayer.size()), 
                ModelSplitter(_loadModel, *_clusterTable, direction));
        } else { // dummy instance
            return new PartitionRecursiveBisection(
                CoordBox<2>(Coord<2>(0, 0), Coord<2>(_width, _height)), 
                Nodes::firstNum(1), 
                Splitter(DVec(1, 1)));
        }
    }


    void repartitionIfNecessary()
    {
        bool necessary = false;
        Partition* oldPartition = _partition;
        Partition* newPartition = createNewPartition();

        double expectedCost = 0; 
        double expectedGain = 0; 

        if (_rank == _master) {
            expectedCost = _communicationModel->predictRepartitionTime(
                        *oldPartition, *newPartition);

            unsigned remainingBalanceChecks = 
                (this->initializer->maxSteps() - this->stepNum) 
                / _loadBalancingEveryN;
            expectedGain = _loadModel->expectedGainFromPartitioning(
                        *oldPartition, *newPartition, remainingBalanceChecks);

            necessary = (expectedGain > expectedCost);
        }
        necessary = _mpilayer.broadcast(necessary, _master);

        if (necessary) {
            _repartitionTimer.tic();

            newPartition->copyFromBroadcast(_mpilayer, _master);
            repartition(oldPartition, newPartition);
            _loadModel->repartition(*newPartition);

            _repartitionTimer.toc();
            long long cycleLength, workLength;
            _repartitionTimer.nextCycle(&cycleLength, &workLength);
            DVec workLengths = _mpilayer.allGather((double)workLength);

            if (_rank == _master) {
                _communicationModel->addObservation(
                        *oldPartition, *newPartition, workLengths);

                if (_verbose) {
                    std::cout 
                        << "Repartitioning in step " << this->stepNum << "\n"
                        << "expected time: " << expectedCost << "\n"
                        << "observed times: "  << workLengths << "\n\n";
                }
            }
            delete oldPartition;

        } else {
            if (_rank == _master && _verbose) {
                std::cout 
                    << "No repartitioning in step " << this->stepNum << "\n"
                    << "expected cost: " << expectedCost << "\n"
                    << "expected gain: "  << expectedGain << "\n\n";
            }

            delete newPartition;
        }
    }


    void repartition(Partition* oldPartition, Partition* newPartition)
    {
        CoordBox<2> oldSimulationRectangle = _simulationRectangle;
        CoordBox<2> oldPartitionRectangle = _partitionRectangle;

        _partition = newPartition;
        initializeRim(newPartition);

        delete _newGrid;
        _newGrid = new DisplacedGrid<CELL_TYPE>(_simulationRectangle);
        _newGrid->getEdgeCell() = _curGrid->getEdgeCell();

        // create send map
        UCVMap  sendCoords;
        std::set<unsigned> recipients;
        for (CoordBoxSequence<2> s = oldPartitionRectangle.sequence();
                s.hasNext();) {
            Coord<2> coord = s.next();
            unsigned dest = newPartition->nodeForCoord(coord);
            if (dest == _rank) { // don't send to self, just copy
                (*_newGrid)[coord] = (*_curGrid)[coord];
            } else {
                sendCoords[dest].push_back(coord); 
            }
        }

        // create receive map
        UCVMap recvCoords;
        for (CoordBoxSequence<2> s = _partitionRectangle.sequence();
                s.hasNext();) {
            Coord<2> coord = s.next();
            unsigned src = oldPartition->nodeForCoord(coord);
            if (src != _rank) // don't receive from self, we copied
                recvCoords[src].push_back(coord);
        }

        send(_curGrid, createJobsFromCoords(sendCoords, *_curGrid));
        receive(_newGrid, createJobsFromCoords(recvCoords, *_newGrid));
        finishCommunication();

        _recvOuterRimJobs = createJobsFromCoords(
                recvOuterRimMap(newPartition), *_newGrid);
        _sendInnerRimJobs = createJobsFromCoords(
                sendInnerRimMap(newPartition), *_newGrid);

        send(_newGrid, _sendInnerRimJobs);
        receive(_newGrid, _recvOuterRimJobs);
        finishCommunication();

        delete _curGrid;
        _curGrid = new DisplacedGrid<CELL_TYPE>(_simulationRectangle);
        _curGrid->getEdgeCell() = _newGrid->getEdgeCell();
        std::swap(_curGrid, _newGrid);
    }


    void summary()
    {
        _reportStream 
            << timerSummary("load balancing", _balanceTimer)
            << timerSummary("updating", _totalUpdateTimer)
            << timerSummary("starting sends", _totalSendTimer)
            << timerSummary("starting recvs", _totalRecvTimer)
            << timerSummary("waiting for send/recv to complete", _totalWaitTimer)
            << _loadModel->summary();
    }


    std::string timerSummary(const std::string& description, Chronometer& timer)
    {
        long long cycleLength, workLength;
        timer.nextCycle(&cycleLength, &workLength);
        std::stringstream result;
        result
            << "Simulator at rank " << _rank << " spent " 
            << (double)workLength / (double)cycleLength
            << " of its time with " << description << ".\n";
        return result.str();
    }
};

};

#endif
#endif
