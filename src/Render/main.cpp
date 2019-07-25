#include <iostream>
#include <opencv2/opencv.hpp>

extern "C" {

void VisualizeDirection(const char* output_file, float* color, cv::Point2f* Qx, cv::Point2f* Qy, int height, int width) {
	cv::Mat final(height, width, CV_32FC3);
	memcpy(final.data, color, sizeof(float) * 3 * height * width);
	final.convertTo(final, CV_8UC3, 255);
	cv::cvtColor(final, final, CV_RGB2BGR);

	for (int i = 0; i < 300; ++i) {
		int px = rand() % width;
		int py = rand() % height;

		for (int k = 0; k < 2; ++k) {
			std::vector<cv::Point2f> points;
			points.push_back(cv::Point2f(px, py));
			for (int j = 0; j < 300; ++j) {
				auto p = points.back();
				int px = p.x;
				int py = p.y;
				if (px < 0 || py < 0 || px >= width - 2 || py >= height - 2)
					break;
				auto& Q = (k == 0) ? Qx : Qy;
				auto p11 = Q[py * width + px];
				auto p12 = Q[py * width + px + 1];
				auto p21 = Q[py * width + px + width];
				auto p22 = Q[py * width + px + width + 1];
				if (p11.dot(p11) < 1e-6)
					break;
				if (p12.dot(p12) < 1e-6)
					break;
				if (p21.dot(p21) < 1e-6)
					break;
				if (p22.dot(p22) < 1e-6)
					break;

				float wx = p.x - px;
				float wy = p.y - py;
				cv::Point2f dp = (p11 * (1 - wx) + p12 * wx) * (1 - wy) + (p21 * (1 - wx) + p22 * wx) * wy;
				points.push_back(p + dp);
				if (k == 0) {
					cv::line(final, p, points.back(), cv::Scalar(255, 0, 0));
				} else {
					cv::line(final, p, points.back(), cv::Scalar(0, 255, 0));
				}
			}
		}
	}
	cv::imwrite(output_file, final);
}


};