#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "puzzle_ocr.h"

int main(int argc, char** argv)
{
	PuzzleOCR ocr;
	ocr.Load(argv[1]);
	ocr.ExtractData();
	ocr.RobustifyConnectivity();
	ocr.ComputeConnectedComponents();
	ocr.ComputeGridGraph();

	auto info = ocr.ExtractFields();
	for (int i = 0; i < info.size(); ++i) {
		auto f = info[i];
		for (int y = 0; y < f.size(); ++y) {
			for (int x = 0; x < f[0].size(); ++x) {
				if (f[y][x].rows == 0) printf("#");
				else printf(".");
			}
			puts("");
		}
		puts("---------------------------------");
	}

	ocr.Show();

	return 0;
}
