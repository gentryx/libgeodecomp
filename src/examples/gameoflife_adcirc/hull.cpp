#include <libgeodecomp/communication/typemaps.h>
#include <libgeodecomp/communication/mpilayer.h>
#include <libgeodecomp/parallelization/serialsimulator.h>
#include <libgeodecomp/parallelization/stripingsimulator.h>
#include <libgeodecomp/storage/multicontainercell.h>
#include <libgeodecomp/config.h>
#include <libgeodecomp/geometry/stencils.h>
#include <libgeodecomp/io/bovwriter.h>
#include <libgeodecomp/io/ppmwriter.h>
#include <libgeodecomp/io/simplecellplotter.h>
#include <libgeodecomp/io/simpleinitializer.h>
#include <libgeodecomp/io/tracingwriter.h>
#include <libgeodecomp/loadbalancer/oozebalancer.h>
#include <libgeodecomp/loadbalancer/tracingbalancer.h>
#include <libgeodecomp/misc/apitraits.h>
#include <libgeodecomp/storage/image.h>


using namespace LibGeoDecomp;

double cross(const FloatCoord<2> &o, const FloatCoord<2> &a, const FloatCoord<2> &b)
{
    return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0]);
}


bool floatCoordCompare(const FloatCoord<2> &a, const FloatCoord<2> &b)
{
    return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]);
};

std::vector<FloatCoord<2> > convexHull(std::vector<FloatCoord<2> > *points)
{
    int k = 0;
    std::vector<FloatCoord<2> > points_sorted = *points;
    std::vector<FloatCoord<2> > hull(2*points_sorted.size());
    int leftMostID=0;
    
    std::sort(points_sorted.begin(), points_sorted.end(), floatCoordCompare);

    int n = points_sorted.size();

    for (int i=0; i<n; i++){        
        while (k >= 2 && cross(hull[k-2], hull[k-1], points_sorted[i]) <= 0) k--;
        hull[k++] = points_sorted[i];
    }

    for (int i=n-2, t=k+1; i>=0; i--){
        while (k>=t && cross(hull[k-2], hull[k-1], points_sorted[i]) <=0 ) k--;
        hull[k++]=points_sorted[i];
    }

    hull.resize(k);

    return hull;
}
    
