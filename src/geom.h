#pragma once

#include "point.h"
#include "quadrilateral.h"

namespace geom
{
double SignedTriangleArea(const Point &a, const Point &b, const Point &c);
double QuadrilateralArea(const Quadrilateral &q);
double Distance(const Point &a, const Point &b);
double Angle(const Point &o, const Point &a, const Point &b);
}
