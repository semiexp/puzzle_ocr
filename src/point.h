#pragma once

#include <cmath>

struct Point
{
	Point() : x(0), y(0) {}
	Point(double x, double y) : x(x), y(y) {}

	inline bool operator<(const Point &) const { return false; }
	inline Point operator+(const Point &other) const {
		return Point(x + other.x, y + other.y);
	}
	inline Point operator-(const Point &other) const {
		return Point(x - other.x, y - other.y);
	}
	inline Point operator*(const double r) const {
		return Point(x * r, y * r);
	}
	inline Point operator/(const double r) const {
		return Point(x / r, y / r);
	}
	inline double abs() const {
		return sqrt(x * x + y * y);
	}

	double x, y;
};
