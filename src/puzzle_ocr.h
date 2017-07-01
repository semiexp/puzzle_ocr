#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class PuzzleOCR
{
public:
	void Load(const char *file);
	void Show();

private:
	cv::Mat image_;
};
