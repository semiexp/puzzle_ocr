#include "recognizer.h"

#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <random>

void Recognizer::TrainFromFile(const char *file)
{
	std::ifstream ifs(file);
	int id;

	std::vector<std::vector<bool> > images;
	std::vector<int> label;

	while (ifs >> id) {
		std::vector<std::vector<bool> > data;
		for (int i = 0; i < 32; ++i) {
			std::string line;
			ifs >> line;
			std::vector<bool> line_b;
			for (int j = 0; j < 32; ++j) line_b.push_back(line[j] == '#' ? true : false);
			data.push_back(line_b);
		}
		auto data2 = ExtractLargestComponent(data);
		data = data2;

		std::vector<bool> data_linear;
		for (int i = 0; i < 32; ++i) {
			for (int j = 0; j < 32; ++j) {
				data_linear.push_back(data[i][j]);
			}
		}

		images.push_back(data_linear);
		label.push_back(id);
	}

	cv::Mat train_data_mat(images.size(), 1024, CV_32F);
	cv::Mat train_label_mat(images.size(), 1, CV_32S);
	for (int j = 0; j < images.size(); ++j) {
		for (int k = 0; k < 1024; ++k) {
			train_data_mat.at<float>(j, k) = images[j][k] ? 1.0 : 0.0;
		}
		train_label_mat.at<int>(j, 0) = label[j];
	}

	svm_ = cv::ml::SVM::create();
	svm_->setType(cv::ml::SVM::C_SVC);
	svm_->setKernel(cv::ml::SVM::RBF);
	svm_->setGamma(0.01);
	svm_->train(train_data_mat, cv::ml::ROW_SAMPLE, train_label_mat);
}
void Recognizer::Save(const char *file)
{
	svm_->save(file);
}
void Recognizer::Load(const char *file)
{
	svm_ = cv::Algorithm::load<cv::ml::SVM>(file);
}
int Recognizer::Recognize(const cv::Mat &img)
{
	std::vector<std::vector<bool> > img_vec(img.rows, std::vector<bool>(img.cols, false));
	for (int y = 0; y < img.rows; ++y) {
		for (int x = 0; x < img.cols; ++x) {
			img_vec[y][x] = img.at<uchar>(y, x) == 0 ? true : false;
		}
	}
	img_vec = ExtractLargestComponent(img_vec);
	cv::Mat img_mat(1, 1024, CV_32F);
	for (int i = 0; i < 1024; ++i) {
		img_mat.at<float>(0, i) = img_vec[i / 32][i % 32];
	}

	return svm_->predict(img_mat);
}
std::vector<std::vector<int> > Recognizer::RecognizeAll(const std::vector<std::vector<cv::Mat> > &field)
{
	std::vector<std::vector<int> > ret;
	for (int i = 0; i < field.size(); ++i) {
		std::vector<int> row;
		for (int j = 0; j < field[i].size(); ++j) {
			if (field[i][j].rows > 0) {
				row.push_back(Recognize(field[i][j]));
			} else {
				row.push_back(-1);
			}
		}
		ret.push_back(row);
	}
	return ret;
}
namespace
{
int FillConnectedComponent(int y, int x, const std::vector<std::vector<bool> > &data, std::vector<std::vector<int> > &sto, int c)
{
	if (y < 0 || x < 0 || y >= data.size() || x >= data[0].size() || !data[y][x]) return 0;
	if (sto[y][x] == c) return 0;
	int ret = 1;
	sto[y][x] = c;
	ret += FillConnectedComponent(y - 1, x, data, sto, c);
	ret += FillConnectedComponent(y + 1, x, data, sto, c);
	ret += FillConnectedComponent(y, x - 1, data, sto, c);
	ret += FillConnectedComponent(y, x + 1, data, sto, c);
	return ret;
}
}
std::vector<std::vector<bool> > Recognizer::ExtractLargestComponent(std::vector<std::vector<bool> > data)
{
	std::vector<std::vector<int> > grp_id(data.size(), std::vector<int>(data[0].size(), -1));
	int id = 0;
	std::pair<int, int> largest{ -1, -1 };
	for (int i = 0; i < data.size(); ++i) {
		for (int j = 0; j < data[0].size(); ++j) {
			if (data[i][j] && grp_id[i][j] == -1) {
				int cnt = FillConnectedComponent(i, j, data, grp_id, id);
				largest = std::max(largest, { cnt, id });
				++id;
			}
		}
	}

	std::vector<std::vector<bool> > ret = data;
	for (int i = 0; i < data.size(); ++i) {
		for (int j = 0; j < data[0].size(); ++j) {
			if (grp_id[i][j] != largest.second) ret[i][j] = false;
		}
	}
	return ret;
}
