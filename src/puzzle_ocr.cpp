#include "puzzle_ocr.h"

#include <vector>
#include <algorithm>
#include <queue>

#include "point.h"
#include "quadrilateral.h"
#include "geom.h"

void PuzzleOCR::Load(const char *file)
{
	image_ = cv::imread(file, cv::IMREAD_GRAYSCALE);
	cv::adaptiveThreshold(image_, image_, 255, CV_ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 31, 7);
}

void PuzzleOCR::Show()
{
	cv::Mat tmp;
	cv::resize(image_, tmp, cv::Size(), 1, 1);
	cv::imshow("View", tmp);

	cv::waitKey();
}

void PuzzleOCR::ExtractData()
{
	int height = image_.rows, width = image_.cols;

	data_ = std::vector<std::vector<bool> >(height, std::vector<bool>(width, false));
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			data_[y][x] = image_.at<uchar>(y, x) == 0 ? 1 : 0;
		}
	}
}

void PuzzleOCR::RobustifyConnectivity()
{
	const int neighbor_size = 9;
	const int dy[] = { -1, 0, 1, 0 }, dx[] = { 0, -1, 0, 1 };
	const double PI = acos(-1);

	int height = image_.rows, width = image_.cols;
	int is_connected[2 * neighbor_size + 1][2 * neighbor_size + 1];

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			if (!data_[y][x]) continue;

			int dy_min = std::max(-y, -neighbor_size);
			int dy_max = std::min(neighbor_size, height - y - 1);
			int dx_min = std::max(-x, -neighbor_size);
			int dx_max = std::min(neighbor_size, width - x - 1);

			for (int dy = dy_min; dy <= dy_max; ++dy) {
				for (int dx = dx_min; dx <= dx_max; ++dx) {
					is_connected[dy + neighbor_size][dx + neighbor_size] = data_[y + dy][x + dx] ? 1 : 0;
				}
			}

			std::queue<std::pair<int, int> > Q;
			Q.push({ 0, 0 });
			is_connected[neighbor_size][neighbor_size] = 2;

			while (!Q.empty()) {
				auto p = Q.front(); Q.pop();
				for (int d = 0; d < 4; ++d) {
					int dy2 = p.first + dy[d], dx2 = p.second + dx[d];
					if (-neighbor_size <= dy2 && dy2 <= neighbor_size && -neighbor_size <= dx2 && dx2 <= neighbor_size && is_connected[dy2 + neighbor_size][dx2 + neighbor_size] == 1) {
						is_connected[dy2 + neighbor_size][dx2 + neighbor_size] = 2;
						Q.push({ dy2, dx2 });
					}
				}
			}

			int dy_min2 = std::max(dy_min, -(neighbor_size / 2));
			int dy_max2 = std::min(dy_max, neighbor_size / 2);
			int dx_min2 = std::max(dx_min, -(neighbor_size / 2));
			int dx_max2 = std::min(dx_max, neighbor_size / 2);

			int outsider_dy = neighbor_size, outsider_dx = 0;
			for (int dy = dy_min2; dy <= dy_max2; ++dy) {
				for (int dx = dx_min2; dx <= dx_max2; ++dx) {
					if (is_connected[dy + neighbor_size][dx + neighbor_size] == 1) {
						if (outsider_dy * outsider_dy + outsider_dx * outsider_dx > dy * dy + dx * dx) {
							outsider_dy = dy;
							outsider_dx = dx;
						}
					}
				}
			}

			if (outsider_dy == neighbor_size) continue;

			double average_angle = 0;
			int n_units = 0;
			for (int dy = dy_min; dy <= dy_max; ++dy) {
				for (int dx = dx_min; dx <= dx_max; ++dx) {
					if (is_connected[dy + neighbor_size][dx + neighbor_size] == 2 && (dy != 0 || dx != 0)) {
						++n_units;
						average_angle += geom::Angle(Point(0, 0), Point(dx, dy), Point(outsider_dx, outsider_dy));
					}
				}
			}

			if (n_units < 10) continue;
			average_angle /= n_units;
			if (average_angle > PI * 0.75) {
				int len = std::max(abs(outsider_dx), abs(outsider_dy));
				for (int d = 0; d <= len; ++d) {
					data_[y + outsider_dy * d / len][x + outsider_dx * d / len] = true;
					image_.at<uchar>(y + outsider_dy * d / len, x + outsider_dx * d / len) = 0;
				}
			}
		}
	}
}

void PuzzleOCR::ComputeConnectedComponents()
{
	const int dy[] = { -1, 0, 1, 0 }, dx[] = { 0, -1, 0, 1 };

	int height = image_.rows, width = image_.cols;

	std::vector<std::vector<bool> > visited(height, std::vector<bool>(width, false));
	components_.clear();

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			if (!data_[y][x] && !visited[y][x]) {
				std::pair<int, Point> ul, ur, dl, dr;
				ul = std::make_pair(y + x, Point(x, y));
				ur = std::make_pair(y - x, Point(x, y));
				dl = std::make_pair(-y + x, Point(x, y));
				dr = std::make_pair(-y - x, Point(x, y));

				std::queue<std::pair<int, int> > qu;
				qu.push({ y, x });
				visited[y][x] = true;

				while (!qu.empty()) {
					auto pt = qu.front(); qu.pop();
					
					for (int d = 0; d < 4; ++d) {
						int y2 = pt.first + dy[d], x2 = pt.second + dx[d];
						if (0 <= y2 && y2 < height && 0 <= x2 && x2 < width && !data_[y2][x2] && !visited[y2][x2]) {
							ul = std::min(ul, std::make_pair(y2 + x2, Point(x2, y2)));
							ur = std::min(ur, std::make_pair(y2 - x2, Point(x2, y2)));
							dl = std::min(dl, std::make_pair(-y2 + x2, Point(x2, y2)));
							dr = std::min(dr, std::make_pair(-y2 - x2, Point(x2, y2)));
							visited[y2][x2] = true;
							qu.push({ y2, x2 });
						}
					}
				}

				Quadrilateral qr;
				qr.ul = ul.second;
				qr.ur = ur.second;
				qr.dl = dl.second;
				qr.dr = dr.second;

				if (geom::QuadrilateralArea(qr) > 0.05 * height * width) continue;

				components_.push_back(qr);
			}
		}
	}
}

void PuzzleOCR::ComputeGridGraph()
{
	cells_ = std::vector<GridCell>(components_.size());

	for (int i = 0; i < components_.size(); ++i) {
		for (int j = i + 1; j < components_.size(); ++j) {
			Quadrilateral qi = components_[i], qj = components_[j];
			double area_i = geom::QuadrilateralArea(qi), area_j = geom::QuadrilateralArea(qj);

			if (1.2 * std::min(area_i, area_j) < std::max(area_i, area_j)) continue;

			double allowed_distance = sqrt(std::min(area_i, area_j)) * 0.2;

			if (geom::Distance(qi.ul, qj.dl) < allowed_distance && geom::Distance(qi.ur, qj.dr) < allowed_distance) {
				cells_[i].up = j;
				cells_[j].down = i;
			}
			if (geom::Distance(qi.dl, qj.ul) < allowed_distance && geom::Distance(qi.dr, qj.ur) < allowed_distance) {
				cells_[i].down = j;
				cells_[j].up = i;
			}
			if (geom::Distance(qi.ul, qj.ur) < allowed_distance && geom::Distance(qi.dl, qj.dr) < allowed_distance) {
				cells_[i].left = j;
				cells_[j].right = i;
			}
			if (geom::Distance(qi.ur, qj.ul) < allowed_distance && geom::Distance(qi.dr, qj.dl) < allowed_distance) {
				cells_[i].right = j;
				cells_[j].left = i;
			}
		}
	}

	std::vector<bool> vis(components_.size(), false);
	std::vector<int> next_ids(components_.size(), -1);
	int next_id_last = 0;

	for (int i = 0; i < components_.size(); ++i) {
		if (vis[i]) continue;

		std::queue<int> Q;
		Q.push(i);
		vis[i] = true;

		std::vector<int> cur;

		while (!Q.empty()) {
			int p = Q.front(); Q.pop();
			cur.push_back(p);

			int cand[] = { cells_[p].up, cells_[p].left, cells_[p].right, cells_[p].down };
			for (int q : cand) {
				if (q != -1 && !vis[q]) {
					vis[q] = true;
					Q.push(q);
				}
			}
		}

		if (cur.size() > 16) {
			for (int p : cur) {
				next_ids[p] = next_id_last++;
			}
		}
	}

	std::vector<Quadrilateral> components_new(next_id_last);
	std::vector<GridCell> cells_new(next_id_last);

	for (int i = 0; i < components_.size(); ++i) {
		if (next_ids[i] != -1) {
			int ni = next_ids[i];

			components_new[ni] = components_[i];
			cells_new[ni] = cells_[i];

			if (cells_new[ni].up != -1) cells_new[ni].up = next_ids[cells_new[ni].up];
			if (cells_new[ni].left != -1) cells_new[ni].left = next_ids[cells_new[ni].left];
			if (cells_new[ni].right != -1) cells_new[ni].right = next_ids[cells_new[ni].right];
			if (cells_new[ni].down != -1) cells_new[ni].down = next_ids[cells_new[ni].down];
		}
	}

	std::swap(components_, components_new);
	std::swap(cells_, cells_new);

	for (auto qr : components_) {
		cv::line(image_, cv::Point(qr.ul.x, qr.ul.y), cv::Point(qr.ur.x, qr.ur.y), 127, 2);
		cv::line(image_, cv::Point(qr.dl.x, qr.dl.y), cv::Point(qr.dr.x, qr.dr.y), 127, 2);
		cv::line(image_, cv::Point(qr.ul.x, qr.ul.y), cv::Point(qr.dl.x, qr.dl.y), 127, 2);
		cv::line(image_, cv::Point(qr.ur.x, qr.ur.y), cv::Point(qr.dr.x, qr.dr.y), 127, 2);
	}
}
