#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

#include "quadrilateral.h"

class PuzzleOCR
{
public:
	void Load(const char *file);
	void Show();

	void ExtractData();
	void RobustifyConnectivity();
	void ComputeConnectedComponents();
	void ComputeGridGraph();

private:
	struct GridCell
	{
		int up, left, right, down;

		GridCell() : up(-1), left(-1), right(-1), down(-1) {}
	};

	cv::Mat image_;
	std::vector<std::vector<bool> > data_;
	std::vector<Quadrilateral> components_;
	std::vector<GridCell> cells_;

};
