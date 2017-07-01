#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "puzzle_ocr.h"

int main(int argc, char** argv)
{
	PuzzleOCR ocr;
	ocr.Load(argv[1]);
	ocr.Show();

	return 0;
}
