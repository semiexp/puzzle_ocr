#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <vector>
#include <string>

class Recognizer
{
public:
	void TrainFromFile(const char *file);
	void Save(const char *file);
	void Load(const char *file);
	int Recognize(const cv::Mat &img);
	std::vector<std::vector<int> > RecognizeAll(const std::vector<std::vector<cv::Mat> > &field);
	std::vector<std::vector<bool> > ExtractLargestComponent(std::vector<std::vector<bool> > data);

private:
	cv::Ptr<cv::ml::SVM> svm_;
};
