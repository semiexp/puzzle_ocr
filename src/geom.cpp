#include "geom.h"

#include <cmath>

namespace geom
{
double SignedTriangleArea(const Point &a, const Point &b, const Point &c)
{
	return ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) / 2;
}
double QuadrilateralArea(const Quadrilateral &q)
{
	return fabs(SignedTriangleArea(q.ul, q.ur, q.dr) + SignedTriangleArea(q.ul, q.dr, q.dl));
}
double Distance(const Point &a, const Point &b)
{
	return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}
}
