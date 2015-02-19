#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <libgeodecomp.h>

#include <libgeodecomp/communication/typemaps.h>
#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>

#include <libgeodecomp/storage/multicontainercell.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <math.h>

#include <boost/assign/std/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

#include <libgeodecomp/config.h>
#include <libgeodecomp/geometry/stencils.h>
#include <libgeodecomp/geometry/partitions/zcurvepartition.h>
#include <libgeodecomp/io/bovwriter.h>
#include <libgeodecomp/io/ppmwriter.h>
#include <libgeodecomp/io/simplecellplotter.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/loadbalancer/oozebalancer.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/storage/image.h>

#include "hull.h"

using namespace boost::assign;
using namespace LibGeoDecomp;

FloatCoord<2> origin;
FloatCoord<2> quadrantDim;


extern "C"{
    void avgdepth_(int *numnodes, float depth[], float *avg);
}


struct neighbor
{
    int neighborID;
    std::vector<int> sendNodes;
    std::vector<int> recvNodes;

    template <class ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & neighborID & sendNodes & recvNodes;
    }

};

struct neighborTable
{
    std::vector<neighbor> myNeighbors;

    template <class ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & myNeighbors;
    }

};

struct ownerTableEntry
{
    int globalID;
    int localID;
    int ownerID;

    template <class ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & globalID & localID & ownerID;
    }

};

struct element
{
    std::vector<int> nodeIDs;

    template <class ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & nodeIDs;
    }
};


struct SubNode
{
    int globalID;
    int localID;
    int alive;
    int lastAlive;
    FloatCoord<3> location;
    std::vector<int> neighboringNodes;

    template <class ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & globalID & localID & alive & lastAlive & location & neighboringNodes;
    }
};


// Each instance represents one subdomain of an ADCIRC unstructured grid
class DomainCell
{
public:
    class API:
        public::APITraits::HasCustomRegularGrid,
            public::APITraits::HasUnstructuredGrid,
            public::APITraits::HasPointMesh
    {
    public:
        inline FloatCoord<2> getRegularGridSpacing()
        {
            return quadrantDim;
        }

        inline FloatCoord<2> getRegularGridOrigin()
        {
            return origin;
        }
    };

    template <class ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & center & id & alive & outputStep & neighboringNodes & myNeighborTable & localNodes & outputStep;
        /*
          LibGeoDecomp::FloatCoord<2> center; // Coordinates of the center
          // of the Domain
          int id; // ID of the domain
          int alive;

          int outputStep;

          std::vector<int> neighboringNodes;   //IDs of neighboring nodes
          neighborTable myNeighborTable;

          std::map<int,SubNode> localNodes;
        */
    }

    DomainCell(const LibGeoDecomp::FloatCoord<2>& center = FloatCoord<2>(), int id = 0, int outputStep = 0) :
        center(center), // Coordinates of the center of the domain
        id(id),         // Integer ID of the domain
        outputStep(outputStep)
    {}

    template<typename NEIGHBORHOOD>
    void update(const NEIGHBORHOOD& hood, int nanoStep);

    std::vector<SubNode> getBoundaryNodes(const int domainID) const
    {
        std::vector<int> outgoingNodeIDs;
        std::vector<SubNode> outgoingNodes;
        DomainCell domainCell = *this;
        for (std::size_t i=0; i<domainCell.myNeighborTable.myNeighbors.size(); i++)
            {
                if (domainCell.myNeighborTable.myNeighbors[i].neighborID == domainID)
                    {
                        for (std::size_t j=0; j<domainCell.myNeighborTable.myNeighbors[i].sendNodes.size(); j++)
                            {
                                outgoingNodeIDs.push_back(domainCell.myNeighborTable.myNeighbors[i].sendNodes[j]);
                            }
                    }

            }

        for (std::size_t i=0; i<outgoingNodeIDs.size(); i++)
            {
                outgoingNodes.push_back(domainCell.localNodes[outgoingNodeIDs[i]]);
            }

        return outgoingNodes;
    }

    void pushNeighborNode(const int neighborID)
    {
        if (std::count(neighboringNodes.begin(), neighboringNodes.end(), neighborID) == 0) {
            neighboringNodes << neighborID;
        }
    }

    void pushNeighborTable(const neighborTable myNeighborTable)
    {
        this->myNeighborTable = myNeighborTable;
    }


    void pushLocalNode(const SubNode resNodeID)
    {
        this->localNodes[resNodeID.localID]=resNodeID;
    }


    const LibGeoDecomp::FloatCoord<2>& getPoint() const
    {
        return center;
    }

    std::vector<LibGeoDecomp::FloatCoord<2> > getShape() const
    {
        std::vector<FloatCoord<2> > points;
        std::vector<FloatCoord<2> > ret;
        //Move localNode locations into a vector of FloatCoord<2>'s
        for (std::map<int, SubNode>::const_iterator i=localNodes.begin(); i!=localNodes.end(); ++i)
        {
            FloatCoord<2> point;
            point[0]=i->second.location[0];
            point[1]=i->second.location[1];

            // If statement makes only resident nodes count for the shape
            if (i->second.globalID != -1) {
                points.push_back(point);
            }
        }

        ret = convexHull(&points);

        return ret;
    }

    LibGeoDecomp::FloatCoord<2> center; // Coordinates of the center
                                        // of the Domain
    int id; // ID of the domain
    int alive;

    int outputStep;

    std::vector<int> neighboringNodes;   //IDs of neighboring nodes
    neighborTable myNeighborTable;

    std::map<int,SubNode> localNodes;


};

// ContainerCell translates between the unstructured grid and the
// regular grid currently required by LGD
typedef ContainerCell<DomainCell, 1000> ContainerCellType;

// type definition required for callback functions below
typedef LibGeoDecomp::TopologiesHelpers::Topology<2, false, false, false > TopologyType;
typedef LibGeoDecomp::Grid<ContainerCellType, TopologyType> GridType;
typedef LibGeoDecomp::CoordMap<ContainerCellType, GridType> BaseNeighborhood;
typedef LibGeoDecomp::NeighborhoodAdapter<BaseNeighborhood, 2> Neighborhood;

const Neighborhood *neighborhood;
DomainCell *domainCell;
template<typename NEIGHBORHOOD>
void DomainCell::update(const NEIGHBORHOOD& hood, int nanoStep)
{
    domainCell = this;
    neighborhood = &hood;
    int domainID = domainCell->id;
    int numNeighbors = myNeighborTable.myNeighbors.size();

    std::cerr << "domainID = " << domainID << " step = " << outputStep << std::endl;

    if (outputStep == 0) {

        //Initial Output
        std::ostringstream filename;
        filename << "data/output" << domainID << "." << nanoStep << ".dat";
        std::ofstream file(filename.str().c_str());
        for (std::map<int, SubNode>::const_iterator i=localNodes.begin(); i!=localNodes.end(); ++i) {
            if (i->second.globalID != -1) {
                file << i->second.localID << " ";
                file << i->second.location[0] << " ";
                file << i->second.location[1] << " ";
                file << i->second.alive << std::endl;
            }
        }
        file.close();
    }

    //FORTRAN interface testing
    {
      int numnodes = 0;
        for (std::map<int, SubNode>::const_iterator i=localNodes.begin(); i!=localNodes.end(); ++i) {
            if (i->second.globalID != -1) {
	      int localID = i->second.localID;
              float xlocation = i->second.location[0];
              float ylocation = i->second.location[1];
              int alive = i->second.alive;
	      numnodes++;
            }
        }
      float depth[numnodes];
      int count = 0;
      for (std::map<int, SubNode>::const_iterator i=localNodes.begin(); i!=localNodes.end(); ++i) {
	  if (i->second.globalID != -1) {
	      depth[count++] = i->second.location[2];
          }
      }

      float avg = 0.0;
      avgdepth_(&numnodes, depth, &avg);
      std::cout << "average depth = " << avg << std::endl;
    }


     //Debugging
    /*
    std::cerr << "I am domain number " << domainID << ", ";
    //Loop over neighbors
    std::cerr << "and I have " << numNeighbors << " neighbors.\n";
    std::cerr << "They are:\n";
    for (int i=0; i<numNeighbors; i++)
    {
        std::cerr << "  ";
        std::cerr << domainCell->myNeighborTable.myNeighbors[i].neighborID;
        std::cerr << ", to whom I will send: ";
        for (int j=0; j<domainCell->myNeighborTable.myNeighbors[i].sendNodes.size(); j++)
        {
            std::cerr << domainCell->myNeighborTable.myNeighbors[i].sendNodes[j] << " ";
        }
        std::cerr << ", and from whom I will receive: ";
        for (int j=0; j<domainCell->myNeighborTable.myNeighbors[i].recvNodes.size(); j++)
        {
            std::cerr << domainCell->myNeighborTable.myNeighbors[i].recvNodes[j] << " ";
        }
        std::cerr << "\n";
    }
    std::cerr << "\n";
    */

    for (std::map<int, SubNode>::iterator i=localNodes.begin(); i!=localNodes.end(); ++i) {
        i->second.lastAlive = i->second.alive;
    }


    //Exchange boundary values
    //Loop over neighbors
    for (int i=0; i<numNeighbors; i++) {
        std::vector<SubNode> incomingNodes;
        int neighborID = myNeighborTable.myNeighbors[i].neighborID;
        incomingNodes = neighborhood[0][neighborID].getBoundaryNodes(domainID);

        // Verify we get the number of nodes we think we should be getting
        if (incomingNodes.size() != myNeighborTable.myNeighbors[i].recvNodes.size()) {
            throw std::runtime_error("boundary node number mismatch!");
        }

        // Verify each node has the same location
        for (int j=0; j<incomingNodes.size(); j++) {
            int myLocalID = myNeighborTable.myNeighbors[i].recvNodes[j];
            FloatCoord<3> localCoords = localNodes[myLocalID].location;
            FloatCoord<3> remoteCoords = incomingNodes[j].location;
            if (localCoords != remoteCoords) {
                throw std::runtime_error("boundary node location mismatch!");
            }

            localNodes[myLocalID].lastAlive = incomingNodes[j].alive;
        }
    }

//    std::vector<FloatCoord<2> > points = getShape();

    //TODO: Interact with a C-style kernel subroutine in another file

    //Simple Kernel
    //Loop over local points
    for (std::map<int, SubNode>::iterator i=localNodes.begin(); i!=localNodes.end(); ++i) {
        int my_alive = i->second.lastAlive;
        if (my_alive == 1) {
            //In this kernel, if the cell was alive previously, it
            //will die the next timestep.
            my_alive = 0;
        } else {
            //Count up neighbor alives
            int sum = 0;
//            std::cerr << "id = " << i->second.localID << ", neighbors =";
            for (int j=0; j<i->second.neighboringNodes.size(); j++) {
                sum += localNodes[i->second.neighboringNodes[j]].lastAlive;
//                std::cerr << " " << i->second.neighboringNodes[j];
           }
//            std::cerr << ", sum = " << sum << std::endl;
            if (sum > 0) {
                my_alive = 1;
            }
        }

        i->second.alive = my_alive;
    }

    //Output
    std::ostringstream filename;
    filename << "data/output" << domainID << "." << outputStep+1 << ".dat";
    std::ofstream file(filename.str().c_str());
    for (std::map<int, SubNode>::const_iterator i=localNodes.begin(); i!=localNodes.end(); ++i) {
        if (i->second.globalID != -1) {
            file << i->second.localID << " ";
            file << i->second.location[0] << " ";
            file << i->second.location[1] << " ";
            file << i->second.alive << std::endl;
        }
    }

    file.close();
    outputStep++;
}



class ADCIRCInitializer : public SimpleInitializer<ContainerCellType>
{
public:
    typedef GridBase<ContainerCellType, 2> GridType;
    using SimpleInitializer<ContainerCellType>::dimensions;

    ADCIRCInitializer(const std::string& meshDir = "", const int steps = 0) :
	SimpleInitializer<ContainerCellType>(Coord<2>(2, 2), steps),
	meshDir(meshDir) {
        determineGridDimensions();
    }

    virtual void grid(GridType *grid) {
        std::ifstream fort80File;

        int numberOfDomains;
        int numberOfElements;
        int numberOfPoints;

        //Neighbor table stuff
        std::vector<neighborTable> myNeighborTables;
        std::vector<ownerTableEntry> ownerTable;
        //--------------------

        openfort80File(fort80File);
        readfort80(fort80File, &numberOfDomains, &numberOfElements, &numberOfPoints, &ownerTable);

        std::vector<std::vector<int> > neighboringDomains;
        std::vector<FloatCoord<2> > centers;

        // clear grid:
        CoordBox<2> box = grid->boundingBox();
        for (CoordBox<2>::Iterator i = box.begin(); i != box.end(); ++i) {
            grid->set(*i, ContainerCellType());
        }


        // piece together domain node cells:
        for (int i=0; i< numberOfDomains; i++) {
            neighborTable myNeighborTable;
            std::ifstream fort18File;
            int numberOfNeighbors=0;
            openfort18File(fort18File, i);
            myNeighborTable = readfort18(fort18File);
            numberOfNeighbors = myNeighborTable.myNeighbors.size();
            myNeighborTables.push_back(myNeighborTable);

            //Read fort.14 file for each domain
            int numberOfPoints;
            int numberOfElements;
            std::ifstream fort14File;
            openfort14File(fort14File, i);
            readFort14Header(fort14File, &numberOfElements, &numberOfPoints);
            std::vector<FloatCoord<3> > points;
            std::vector<int> localIDs;
            readFort14Points(fort14File, &points, &localIDs, numberOfPoints);
            FloatCoord<2> center = determineCenter(&points);
            center /= numberOfPoints;

            centers.push_back(center);

            // Load neighboringDomains with myNeighborTables
            std::vector<int> neighbors;
            for (int j=0; j<numberOfNeighbors; j++) {
                neighbors.push_back(myNeighborTable.myNeighbors[j].neighborID);
            }
            neighboringDomains.push_back(neighbors);

            int nodeID = i;
            DomainCell node(center, nodeID);

            node.pushNeighborTable(myNeighborTable);

            for (int j=0; j<numberOfNeighbors; j++) {
                node.pushNeighborNode(neighbors[j]);
            }

            // What does this loop do??
            for (std::size_t j=0; j<ownerTable.size(); j++) {
                if (ownerTable[j].ownerID == nodeID) {
                    SubNode thissubnode;
                    thissubnode.location = points[ownerTable[j].localID+1]; //FIXME
                    thissubnode.localID = ownerTable[j].localID;
                    thissubnode.globalID = ownerTable[j].globalID;
                }
            }

            // Read elements
            std::vector<element> myElements = readElements(fort14File, numberOfElements);

            // Loop through all local nodes
            for (std::size_t j=0; j<points.size(); j++) {
                SubNode thissubnode;
                thissubnode.location = points[j];
                thissubnode.localID = localIDs[j];
                thissubnode.globalID = -1;
                thissubnode.alive = 0;

                // Initial Seed
                if ((nodeID == 0) && (thissubnode.localID == 5)) {
                    thissubnode.alive = 1;
                }


                // Loop through all global nodes
                for (std::size_t k=0; k<ownerTable.size(); k++) {
                    // If the global node is owned by the current domain,
                    if (nodeID == ownerTable[k].ownerID) {
                        //Then
                        if (localIDs[j] == ownerTable[k].localID) {
                            thissubnode.globalID = ownerTable[k].globalID;
                        }
                    }
                }
                node.pushLocalNode(thissubnode);
            }

            //Assemble local linking information
            //Loop over all Elements
            for (std::size_t j=0; j<myElements.size(); j++) {
                //std::cerr << "elementid = " << j << std::endl;
                //Loop over nodes associated with the element
                for (std::size_t k=0; k<myElements[j].nodeIDs.size(); k++) {
                    int outer_nodeID = node.localNodes[myElements[j].nodeIDs[k]].localID;
                    //std::cerr << "outer_nodeID = " << outer_nodeID << std::endl;
                    for (std::size_t l=0; l<myElements[k].nodeIDs.size(); l++) {
                        int inner_nodeID = node.localNodes[myElements[j].nodeIDs[l]].localID;
                        //std::cerr << "inner_nodeID = " << inner_nodeID << std::endl;
                        bool notAlreadyThere = (std::count(
                            node.localNodes[inner_nodeID].neighboringNodes.begin(),
                            node.localNodes[inner_nodeID].neighboringNodes.end(),
                            outer_nodeID) == 0);
                        if ( (outer_nodeID != inner_nodeID) && notAlreadyThere) {
                            node.localNodes[inner_nodeID].neighboringNodes.push_back(outer_nodeID);
                            //std::cerr << node.localNodes[inner_nodeID].neighboringNodes;
                        }
                    }
                }
            }

            //Debug
            //Loop over all points
            /*
            std::cerr << "domain = " << node.id << std::endl;
            for (std::map<int, SubNode>::const_iterator i=node.localNodes.begin(); i!=node.localNodes.end(); ++i)
            {
                std::cerr << i->second.localID << " " << i->second.neighboringNodes << std::endl;
            }
            */

            FloatCoord<2> gridCoordFloat = (node.center - minCoord) / quadrantDim;
            Coord<2> gridCoord(gridCoordFloat[0], gridCoordFloat[1]);

            ContainerCellType container = grid->get(gridCoord);
            container.insert(node.id, node);
            grid->set(gridCoord, container);
        }
    }


    template <class ARCHIVE>
    void serialize(ARCHIVE& ar, unsigned)
    {
        ar & boost::serialization::base_object<SimpleInitializer<ContainerCellType> >(*this) & meshDir & maxDiameter & minCoord & maxCoord;
    }

private:
    std::string meshDir;
    double maxDiameter;
    FloatCoord<2> minCoord;
    FloatCoord<2> maxCoord;

    void determineGridDimensions()
    {
        std::ifstream fort80File;

        int numberOfDomains;
        int numberOfElements;
        int numberOfPoints;
        std::vector<ownerTableEntry> ownerTable;
        std::vector<neighborTable> myNeighborTables;
        openfort80File(fort80File);
        readfort80(fort80File, &numberOfDomains, &numberOfElements, &numberOfPoints, &ownerTable);

        std::vector<FloatCoord<2> > centers;

        for (int i=0; i< numberOfDomains; i++) {
            neighborTable myNeighborTable;
            std::ifstream fort18File;
            int numberOfNeighbors=0;
            openfort18File(fort18File, i);
            myNeighborTable = readfort18(fort18File);
            numberOfNeighbors = myNeighborTable.myNeighbors.size();

            myNeighborTables.push_back(myNeighborTable);
            // need to push 'myNeighborTable' to the domain

            //Read fort.14 file for each domain
            int numberOfPoints;
            int numberOfElements;
            std::ifstream fort14File;
            openfort14File(fort14File, i);
            readFort14Header(fort14File, &numberOfElements, &numberOfPoints);
            std::vector<FloatCoord<3> > points;
            std::vector<int> localIDs;
            readFort14Points(fort14File, &points, &localIDs, numberOfPoints);
            FloatCoord<2> center = determineCenter(&points);
            center /= numberOfPoints;
            centers.push_back(center);
        }

        maxDiameter = determineMaximumDiameter(&centers, myNeighborTables);
        minCoord = FloatCoord<2>(
            std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max());
        maxCoord = FloatCoord<2>(
            -std::numeric_limits<double>::max(),
            -std::numeric_limits<double>::max());

        for (int i = 0; i < numberOfDomains; ++i) {
            FloatCoord<2> p(centers[i][0], centers[i][1]);

            minCoord = minCoord.min(p);
            maxCoord = maxCoord.max(p);
        }

        origin = minCoord;
        // add a safety factor for the cell spacing so we can be sure
        // neighboring elements are never more than 1 cell apart in the grid:
        quadrantDim = FloatCoord<2>(maxDiameter * 2.0, maxDiameter * 2.0);
        FloatCoord<2> floatDimensions = (maxCoord - minCoord) / quadrantDim;
        dimensions = Coord<2>(
            ceil(floatDimensions[0]),
            ceil(floatDimensions[1]));


        std::cerr << "geometry summary:\n"
                  << "  minCoord: "    << minCoord    << "\n"
                  << "  maxCoord: "    << maxCoord    << "\n"
                  << "  maxDiameter: " << maxDiameter << "\n"
                  << "  dimensions: "  << dimensions  << "\n"
                  << "\n";
    }


    FloatCoord<2> determineCenter(std::vector<FloatCoord<3> > *points)
    {
        FloatCoord<2> center;
        for (std::size_t i=0; i < points->size(); i++) {
            FloatCoord<2> coord(points[0][i][0],points[0][i][1]);
            center += coord;
        }

        return center;
    }

    double determineMaximumDiameter(
        const std::vector<FloatCoord<2> >* points,
        const std::vector<neighborTable> myNeighborTables)
    {
        double maxDiameter = 0;

        int numPoints = points->size();
        for (int point = 0; point < numPoints; ++point) {
            int numNeighbors = myNeighborTables[point].myNeighbors.size();
            for (int i=0; i<numNeighbors; i++) {
                int neighborID = myNeighborTables[point].myNeighbors[i].neighborID;
                double dist = getDistance(points[0][point],points[0][neighborID]);
                maxDiameter = std::max(maxDiameter,dist);
            }
        }

        return maxDiameter;
    }

    double getDistance(FloatCoord<2> a, FloatCoord<2> b)
    {
        FloatCoord<2> c = a - b;
        return sqrt(c[0] * c[0] + c[1] * c[1]);
    }

    void openfort80File(std::ifstream& meshFile)
    {
        std::string meshFileName = meshDir;
        meshFileName.append("/fort.80");
        meshFile.open(meshFileName.c_str());
        if (!meshFile) {
            throw std::runtime_error("could not open fort.80 file " + meshFileName);
        }
    }

    void readfort80(std::ifstream& meshFile, int *numberOfDomains, int *numberOfElements, int *numberOfPoints, std::vector<ownerTableEntry> *ownerTable)
    {
        std::string buffer(1024, ' ');
        //Discard first 3 lines:
        std::getline(meshFile, buffer);
        std::getline(meshFile, buffer);
        std::getline(meshFile, buffer);

        meshFile >> *numberOfElements;
        meshFile >> *numberOfPoints;

        //Discard rest of line
        std::getline(meshFile, buffer);

        meshFile >> *numberOfDomains;

        //Discard rest of the line:
        std::getline(meshFile, buffer);

        //Discard next 8 lines:
        std::getline(meshFile, buffer);
        std::getline(meshFile, buffer);
        std::getline(meshFile, buffer);
        std::getline(meshFile, buffer);
        std::getline(meshFile, buffer);
        std::getline(meshFile, buffer);
        std::getline(meshFile, buffer);
        std::getline(meshFile, buffer);

        for (int domain = 0; domain < *numberOfDomains; ++domain) {
            int buf;
            int numNodes;
            meshFile >> buf;
            if (buf != domain) {
                throw std::runtime_error("buf does not match domain");
            }
            meshFile >> numNodes;
            //Discard rest of the line
            std::getline(meshFile, buffer);

            for (int node = 0; node < numNodes; ++node) {
                int nodeNum;
                meshFile >> nodeNum;
            }

            // throw away the rest of the line
            std::getline(meshFile, buffer);
        }
        // Throw away another line
        std::getline(meshFile, buffer);
        for (int node = 0; node < *numberOfPoints; node++) {
            ownerTableEntry thisOwnerTableEntry;
            int global_label;
            int local_label;
            int owner;
            meshFile >> global_label;
            meshFile >> owner;
            meshFile >> local_label;

            thisOwnerTableEntry.globalID=global_label;
            thisOwnerTableEntry.ownerID=owner;
            thisOwnerTableEntry.localID=local_label;

            std::getline(meshFile, buffer);//discard rest of line

            ownerTable->push_back(thisOwnerTableEntry);
        }
    }

    void openfort18File(std::ifstream& meshFile, int domainID)
    {
        std::string meshFileName = meshDir;
        meshFileName.append("/PE");
        std::stringstream buf;
        buf.width(4);
        buf.fill('0');
        buf << domainID;
        meshFileName.append(buf.str());
        meshFileName.append("/fort.18");
        meshFile.open(meshFileName.c_str());
        if (!meshFile) {
            throw std::runtime_error("could not open fort.18 file"+meshFileName);
        }
    }

    neighborTable readfort18(std::ifstream& meshFile)
    {
        neighborTable myNeighborTable;
        int numNeighbors;
        std::string buffer(1024, ' ');
        while (buffer != "COMM") {
            meshFile >> buffer;
        }
        meshFile >> buffer; // PE
            if (buffer != "PE"){
                throw std::runtime_error("buffer does not match!");
            }
        meshFile >> numNeighbors;

        for (int i=0; i<numNeighbors; i++) {
            neighbor neighbor;
            int numberOfRecvNodes;
            meshFile >> buffer; //RECV
            if (buffer != "RECV") {
                throw std::runtime_error("buffer does not match!");
            }
            meshFile >> buffer; //PE
            if (buffer != "PE") {
                throw std::runtime_error("buffer does not match!");
            }
            meshFile >> neighbor.neighborID;
            meshFile >> numberOfRecvNodes;
            std::getline(meshFile, buffer);//discard rest of the line

            //Assemble arrays of nodes to be received
            for (int j=0; j<numberOfRecvNodes; j++) {
                int receiveNode;
                meshFile >> receiveNode;
                neighbor.recvNodes.push_back(receiveNode);
            }
            std::getline(meshFile, buffer);//discard rest of the line
            myNeighborTable.myNeighbors.push_back(neighbor);
        }

        for (int i=0; i<numNeighbors; i++) {
            int neighbor;
            int numberOfSendNodes;
            meshFile >> buffer; //SEND
            if (buffer != "SEND"){
                throw std::runtime_error("buffer does not match!");
            }
            meshFile >> buffer; //PE
            if (buffer != "PE"){
                throw std::runtime_error("buffer does not match!");
            }
            meshFile >> neighbor;
            meshFile >> numberOfSendNodes;
            std::getline(meshFile, buffer);//discard rest of the line
            //Assemble arrays of nodes to be sent
            for (int j=0; j<numberOfSendNodes; j++) {
                int sendNode;
                meshFile >> sendNode;
                myNeighborTable.myNeighbors[i].sendNodes.push_back(sendNode);
            }
            std::getline(meshFile, buffer);//discard rest of the line
        }

        return myNeighborTable;
    }

    void openfort14File(std::ifstream& meshFile, int domainID)
    {
        std::string meshFileName = meshDir;
        meshFileName.append("/PE");
        std::stringstream buf;
        buf.width(4);
        buf.fill('0');
        buf << domainID;
        meshFileName.append(buf.str());
        meshFileName.append("/fort.14");
        meshFile.open(meshFileName.c_str());
        if (!meshFile) {
            throw std::runtime_error("could not open fort.14 file "+meshFileName);
        }
    }



    void readFort14Header(std::ifstream& meshFile, int *numberOfElements, int *numberOfPoints)
    {
        std::string buffer(1024, ' ');
        // discard first line, which only holds comments anyway
        std::getline(meshFile, buffer);

        meshFile >> *numberOfElements;
        meshFile >> *numberOfPoints;

        // discard remainder of line
        std::getline(meshFile, buffer);

        if (!meshFile.good()) {
            throw std::logic_error("could not read header");
        }
    }

    void readFort14Points(std::ifstream& meshFile, std::vector<FloatCoord<3> > *points, std::vector<int> *localIDs, const int numberOfPoints)
    {
        std::string buffer(1024, ' ');

        for (int i = 0; i < numberOfPoints; ++i) {
            FloatCoord<3> p;
            int localID;

            meshFile >> localID;
            meshFile >> p[0];
            meshFile >> p[1];
            meshFile >> p[2];
            std::getline(meshFile, buffer);

            *points << p;
            *localIDs << localID;
        }

        if (!meshFile.good()) {
            throw std::runtime_error("could not read points");
        }
    }

    std::vector<element> readElements(
        std::ifstream& meshFile,
        const int numberOfElements)
    {
        std::vector<element> ret;
        if (!meshFile.good()) {
            throw std::runtime_error("could not read elements");
        }
        std::string buffer(1024, ' ');
        for (int elem = 0; elem < numberOfElements; ++elem) {
            element myElement;
            int buf;
            int numNodes;
            meshFile >> buf; // Element number
            meshFile >> numNodes; // will always be 3
            std::vector<int> nodeIDs;
            for (int i = 0; i < numNodes; ++i) {
                int node;
                meshFile >> node;
                nodeIDs.push_back(node);
            }
            std::getline(meshFile, buffer);
            myElement.nodeIDs = nodeIDs;
            ret.push_back(myElement);
        }

        return ret;
    }

};

typedef
HpxSimulator::HpxSimulator<ContainerCellType, ZCurvePartition<2> >
SimulatorType;

LIBGEODECOMP_REGISTER_HPX_SIMULATOR_DECLARATION(
    SimulatorType,
    DomainCellSimulator)
LIBGEODECOMP_REGISTER_HPX_SIMULATOR(
    SimulatorType,
    DomainCellSimulator)

BOOST_CLASS_EXPORT_GUID(ADCIRCInitializer, "ADCIRCInitializer");
BOOST_CLASS_EXPORT_GUID(DomainCell, "DomainCell");

void runSimulation()
{
    // TODO: figure out what the stuff below is
    Coord<2> dim(3, 2);
    std::size_t numCells = 100;
    double minDistance = 100;
    double quadrantSize = 400;
    quadrantDim = FloatCoord<2>(quadrantSize, quadrantSize);

    // Hardcoded link to the directory
    std::string prunedDirname("/home/zbyerly/research/meshes/qah4");

    // Hardcoded number of simulation steps
    int steps = 100;

    //    SerialSimulator<ContainerCellType> sim(


    //sim(new ADCIRCInitializer(prunedDirname, steps));

    /*
    int ioPeriod = 1;
    SiloWriter<ContainerCellType> *writer = new SiloWriter<ContainerCellType>("mesh", *ioPeriod);
    writer->addSelectorForUnstructuredGrid(
        &DomainCell::alive,
        "DomainCell_alive");
    sim.addWriter(writer);
    */
    ADCIRCInitializer *init = new ADCIRCInitializer(prunedDirname, steps);

    SimulatorType sim(
        init,
        1, //overcommitFactor
        new TracingBalancer(new OozeBalancer()),
        10, //balancingPeriod
        1 // ghostZoneWidth
		      );

    sim.run();
}

int hpx_main()
{
    runSimulation();

    return hpx::finalize();
}

int main(int argc, char *argv[])
{
    return hpx::init(argc, argv);
}
