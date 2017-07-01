#include "puzzle_ocr.h"

void PuzzleOCR::Load(const char *file)
{
	image_ = cv::imread(file, cv::IMREAD_GRAYSCALE);
	cv::adaptiveThreshold(image_, image_, 255, CV_ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 31, 15);
}

void PuzzleOCR::Show()
{
	cv::imshow("View", image_);
	cv::waitKey();
}
