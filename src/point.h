#pragma once

struct Point
{
	Point() : x(0), y(0) {}
	Point(double x, double y) : x(x), y(y) {}

	double x, y;

	inline bool operator<(const Point &) const { return false; }
};
