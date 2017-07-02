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
double Angle(const Point &o, const Point &a, const Point &b)
{
	double oa = Distance(o, a), ob = Distance(o, b), ab = Distance(a, b);
	return acos((oa * oa + ob * ob - ab * ab) / (2 * oa * ob));
}
}
