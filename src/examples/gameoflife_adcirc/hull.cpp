#include <algorithm>
#include "hull.h"

double cross(const FloatCoord<2> &o, const FloatCoord<2> &a, const FloatCoord<2> &b)
{
    return
        (a[0] - o[0]) * (b[1] - o[1]) -
        (a[1] - o[1]) * (b[0] - o[0]);
}

bool floatCoordCompare(const FloatCoord<2>& a, const FloatCoord<2>& b)
{
    return a < b;
};

std::vector<FloatCoord<2> > convexHull(std::vector<FloatCoord<2> > *points)
{
    std::size_t k = 0;
    std::vector<FloatCoord<2> > points_sorted = *points;
    std::vector<FloatCoord<2> > hull(2 * points_sorted.size());

    std::sort(points_sorted.begin(), points_sorted.end(), floatCoordCompare);

    std::size_t n = points_sorted.size();

    for (std::size_t i = 0; i < n; ++i) {
        while ((k >= 2) &&
               cross(hull[k-2], hull[k-1], points_sorted[i]) <= 0) {
            --k;
        }

        hull[k++] = points_sorted[i];
    }

    std::size_t t = k + 1;
    for (long i = n - 2; i >= 0; --i) {
        while ((k >= t) && cross(hull[k-2], hull[k-1], points_sorted[i]) <= 0) {
             --k;
        }

        hull[k++] = points_sorted[i];
    }

    hull.resize(k);

    return hull;
}
