#include <opencv2/opencv.hpp>
#include <glm/glm.hpp>
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

extern "C" {

void VisualizeDirection(const char* output_file, float* color, cv::Point2f* Qx, cv::Point2f* Qy, int height, int width) {
	cv::Mat final(height, width, CV_32FC3);
	memcpy(final.data, color, sizeof(float) * 3 * height * width);
	final.convertTo(final, CV_8UC3, 255);

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

void ComputeWarping(int* integer_params, float delta, float* intrinsics, cv::Vec3f* Qx, cv::Vec3f* Qy, float* output_coords){//, float* frameX, float* frameY) {
	int height = integer_params[0];
	int width = integer_params[1];
	int px = integer_params[2];
	int py = integer_params[3];
	int patch_w = integer_params[4];
	cv::Vec3f pt((px - intrinsics[1]) / intrinsics[0], (py - intrinsics[3]) / intrinsics[2], 1);
	int patch_w2 = patch_w * 2 + 1;
	std::vector<cv::Vec2i> coords;
	coords.reserve(patch_w2 * patch_w2);
	coords.push_back(cv::Vec2i(patch_w, patch_w));
	std::vector<int> hash(patch_w2 * patch_w2);
	hash[patch_w * patch_w2 + patch_w] = 1;
	int f = 0;
	while (f < coords.size()) {
		cv::Vec2i& p = coords[f];
		cv::Vec2i dir[4] = {cv::Vec2i(0,1), cv::Vec2i(1,0), cv::Vec2i(0,-1), cv::Vec2i(-1,0)};
		for (int i = 0; i < 4; ++i) {
			cv::Vec2i np = p + dir[i];
			if (np.val[0] >= 0 && np.val[0] < patch_w2 && np.val[1] >= 0 && np.val[1] < patch_w2) {
				if (hash[np.val[1] * patch_w2 + np.val[0]] == 0) {
					hash[np.val[1] * patch_w2 + np.val[0]] = 1;
					coords.push_back(np);
				}
			}
		}
		f += 1;
	}
	for (auto& h : hash) {
		h = 0;
	}

	std::vector<cv::Vec3f> positions(patch_w2 * patch_w2);
	std::vector<cv::Vec2f> positions2d(patch_w2 * patch_w2);
	std::vector<std::pair<cv::Vec3f, cv::Vec3f> > frames(patch_w2 * patch_w2);

	positions[patch_w * patch_w2 + patch_w] = cv::Vec3f((px - intrinsics[1]) / intrinsics[0], (py - intrinsics[3]) / intrinsics[2], 1);
	positions2d[patch_w * patch_w2 + patch_w] = cv::Vec2f(px, py);
	frames[patch_w * patch_w2 + patch_w] = std::make_pair(Qx[py * width + px], Qy[py * width + px]);
	hash[patch_w * patch_w2 + patch_w] = 1;
	auto p1 = Qx[py * width + px];
	auto p2 = Qy[py * width + px];
	for (int i = 0; i < coords.size(); ++i) {
		int index = coords[i].val[1] * patch_w2 + coords[i].val[0];
		if (hash[index])
			continue;
		int top = 0;
		cv::Vec3f slots[8];
		std::pair<cv::Vec3f, cv::Vec3f> new_frame, frame;
		int dx[] = {-1,-1,-1,0,0,1,1,1};
		int dy[] = {-1,0,1,-1,1,-1,0,1};
		for (int j = 0; j < 8; ++j) {
			int tx = coords[i].val[0] + dx[j];
			int ty = coords[i].val[1] + dy[j];
			if (tx < 0 || ty < 0 || tx >= patch_w2 || ty >= patch_w2)
				continue;
			if (hash[ty * patch_w2 + tx] == 0)
				continue;
			frame = frames[ty * patch_w2 + tx];
			slots[top++] = positions[ty * patch_w2 + tx] - delta * frame.first * dx[j] - delta * frame.second * dy[j];
		}
		cv::Vec3f p(0,0,0);
		for (int j = 0; j < top; ++j) {
			p += slots[j];
		}
		p /= (float)top;
		positions[index] = p;
		hash[index] = 1;
		float current_px = p.val[0] / p.val[2] * intrinsics[0] + intrinsics[1];
		float current_py = p.val[1] / p.val[2] * intrinsics[2] + intrinsics[3];
		positions2d[index] = cv::Vec2f(current_px, current_py);
		if (current_px < 0 || current_px + 1 >= width || current_py < 0 || current_py + 1 >= height)
			continue;
		new_frame = std::make_pair(cv::Vec3f(0,0,0), cv::Vec3f(0,0,0));
		for (int dpy = 0; dpy <= 1; ++dpy) {
			float weight_y = (dpy == 0) ? 1 - (current_py - (int)current_py) : (current_py - (int)current_py);
			for (int dpx = 0; dpx <= 1; ++dpx) {
				float weight_x = (dpy == 0) ? 1 - (current_px - (int)current_px) : (current_px - (int)current_px);
				int index = (int(current_py) + dpy) * width + (int(current_px) + dpx);
				cv::Vec3f dir_x = Qx[index];
				cv::Vec3f dir_y = Qy[index];
				float max_dot = -1e30;
				cv::Vec3f dir_p1, dir_p2;
				for (int k = 0; k < 4; ++k) {
					float dot = std::max(dir_x.dot(frame.first), dir_y.dot(frame.second));
					if (dot > max_dot) {
						max_dot = dot;
						dir_p1 = dir_x;
						dir_p2 = dir_y;
					}
					auto temp = dir_x;
					dir_x = -dir_y;
					dir_y = temp;
				}
				new_frame.first += weight_x * weight_y * dir_p1;
				new_frame.second += weight_x * weight_y * dir_p2;
			}
		}

		double l = new_frame.first.dot(new_frame.first);
		if (l > 0)
			new_frame.first /= sqrt(l);
		l = new_frame.second.dot(new_frame.second);
		if (l > 0)
			new_frame.second /= sqrt(l);
		frames[index] = new_frame;
	}
	memcpy(output_coords, positions2d.data(), sizeof(cv::Vec2f) * positions2d.size());
}

float orient2d(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& c)
{
    return (b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x);
}

int min3(float v1, float v2, float v3) {
	return std::min(v1, std::min(v2, v3));
}
int max3(float v1, float v2, float v3) {
	return 0.99999 + std::max(v1, std::max(v2, v3));
}

void rasterize(unsigned char* rgb1, unsigned char* rgb2, unsigned char* rgb3, cv::Point2f& v0, cv::Point2f& v1, cv::Point2f& v2, unsigned char* image, int height, int width, int solid) {
	  // Compute triangle bounding box
    int minX = min3(v0.x, v1.x, v2.x);
    int minY = min3(v0.y, v1.y, v2.y);
    int maxX = max3(v0.x, v1.x, v2.x);
    int maxY = max3(v0.y, v1.y, v2.y);

    // Clip against screen bounds
    minX = std::max(minX, 0);
    minY = std::max(minY, 0);
    maxX = std::min(maxX, width - 1);
    maxY = std::min(maxY, height - 1);

    // Rasterize
    for (int py = minY; py <= maxY; py++) {
        for (int px = minX; px <= maxX; px++) {
            // Determine barycentric coordinates
            cv::Point2f p(px, py);
            float w0 = orient2d(v1, v2, p);
            float w1 = orient2d(v2, v0, p);
            float w2 = orient2d(v0, v1, p);

            // If p is on or inside all edges, render pixel.
            if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
	            float w = 1.0 / (w0 + w1 + w2);
	            w0 *= w;
	            w1 *= w;
	            w2 *= w;
            	unsigned char* target = image + (py * width + px) * 3;
            	unsigned char t[3];
            	for (int j = 0; j < 3; ++j) {
            		float c = rgb1[j] * w0 + rgb2[j] * w1 + rgb3[j] * w2;
            		if (c > 255)
            			c = 255;
            		if (c < 0)
            			c = 0;
            		t[j] = c;
            	}
            	if (t[0] != 0 || t[1] != 0 || t[2] != 0) {
            		if (solid == 1)
	            		for (int j = 0; j < 3; ++j) {
		            		target[j] = t[j];
	            		}
	            	else
	            		for (int j = 0; j < 3; ++j) {
		            		target[j] = target[j] * 0.5 + t[j] * 0.5;
	            		}
            	}
            }
        }
    }
}
void Rasterize(unsigned char* patch, cv::Point2f* coords, int patch_w, unsigned char* image, int height, int width, int solid) {
	for (int i = 0; i < patch_w - 1; ++i) {
		for (int j = 0; j < patch_w - 1; ++j) {
			{
				unsigned char* rgb1 = patch + (i * patch_w + j) * 3;
				unsigned char* rgb2 = patch + (i * patch_w + j + 1) * 3;
				unsigned char* rgb3 = patch + (i * patch_w + j + patch_w) * 3;
				cv::Point2f& coord1 = coords[i * patch_w + j];
				cv::Point2f& coord2 = coords[i * patch_w + j + 1];
				cv::Point2f& coord3 = coords[i * patch_w + j + patch_w];
				rasterize(rgb1, rgb2, rgb3, coord1, coord2, coord3, image, height, width, solid);
			}
			{
				unsigned char* rgb1 = patch + (i * patch_w + j + 1) * 3;
				unsigned char* rgb2 = patch + (i * patch_w + j + 1 + patch_w) * 3;
				unsigned char* rgb3 = patch + (i * patch_w + j + patch_w) * 3;
				cv::Point2f& coord1 = coords[(i * patch_w + j + 1)];
				cv::Point2f& coord2 = coords[i * patch_w + j + 1 + patch_w];
				cv::Point2f& coord3 = coords[i * patch_w + j + patch_w];
				rasterize(rgb1, rgb2, rgb3, coord1, coord2, coord3, image, height, width, solid);
			}
		}
	}
}


float calculateSignedArea2(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) {
	return ((c.x - a.x) * (b.y - a.y) - (b.x - a.x) * (c.y - a.y));
}

glm::vec3 calculateBarycentricCoordinate(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, const glm::vec3& p) {
	float beta_tri = calculateSignedArea2(a, p, c);
	float gamma_tri = calculateSignedArea2(a, b, p);
	float tri_inv = 1.0f / calculateSignedArea2(a, b, c);
	float beta = beta_tri * tri_inv;
	float gamma = gamma_tri * tri_inv;
	float alpha = 1.0 - beta - gamma;
	return glm::vec3(alpha, beta, gamma);
}

bool isBarycentricCoordInBounds(const glm::vec3 barycentricCoord) {
    return barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
           barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
           barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
}

float getZAtCoordinate(const glm::vec3 barycentricCoord, const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) {
	return (barycentricCoord.x * a.z
		+ barycentricCoord.y * b.z
		+ barycentricCoord.z * c.z);
}


void DrawTriangle(glm::vec3* v1, glm::vec3* v2, glm::vec3* v3,
	glm::vec2* t1, glm::vec2* t2, glm::vec2* t3,
	glm::vec3* n1, glm::vec3* n2, glm::vec3* n3,
	unsigned char* tex_image, unsigned char* color_image, float* zbuffer,
	float* intrinsics, int tex_width, int tex_height, int width, int height) {

	glm::vec3 p1 = *v1, p2 = *v2, p3 = *v3;
	if (p1.z < 0.01 || p2.z < 0.01 || p3.z < 0.01)
		return;

	p1.z = 1.0f / p1.z;
	p2.z = 1.0f / p2.z;
	p3.z = 1.0f / p3.z;

	p1.x = p1.x * p1.z;
	p1.y = p1.y * p1.z;
	p2.x = p2.x * p2.z;
	p2.y = p2.y * p2.z;
	p3.x = p3.x * p3.z;
	p3.y = p3.y * p3.z;

	float fx = intrinsics[0];
	float cx = intrinsics[1];
	float fy = intrinsics[2];
	float cy = intrinsics[3];
	int minX = (MIN(p1.x, MIN(p2.x, p3.x)) * fx + cx);
	int minY = (MIN(p1.y, MIN(p2.y, p3.y)) * fy + cy);
	int maxX = (MAX(p1.x, MAX(p2.x, p3.x)) * fx + cx) + 0.999999f;
	int maxY = (MAX(p1.y, MAX(p2.y, p3.y)) * fy + cy) + 0.999999f;

	minX = MAX(0, minX);
	minY = MAX(0, minY);
	maxX = MIN(width, maxX);
	maxY = MIN(height, maxY);


	for (int py = minY; py <= maxY; ++py) {
		for (int px = minX; px <= maxX; ++px) {
			if (px < 0 || px >= width || py < 0 || py >= height)
				continue;

			float x = (px - cx) / fx;
			float y = (py - cy) / fy;
			glm::vec3 baryCentricCoordinate = calculateBarycentricCoordinate(p1, p2, p3, glm::vec3(x, y, 0));

			if (isBarycentricCoordInBounds(baryCentricCoordinate)) {
				int pixel = py * width + px;

				float z = getZAtCoordinate(baryCentricCoordinate, p1, p2, p3);
				int z_quantize = z * 100000;

				int original_z = zbuffer[pixel];

				if (original_z < z_quantize) {
					glm::vec2 tex = *t1 * baryCentricCoordinate.x + *t2 * baryCentricCoordinate.y + *t3 * baryCentricCoordinate.z;
					glm::vec3 normal = *n1 * baryCentricCoordinate.x + *n2 * baryCentricCoordinate.y + *n3 * baryCentricCoordinate.z;

					glm::vec3 light_dir((px - cx) / fx, (py - cy) / fy, 1);
					light_dir = glm::normalize(light_dir);
					normal = glm::normalize(normal);
					while (tex.x > 1)
						tex.x -= 1;
					while (tex.x < 0)
						tex.x += 1;
					while (tex.y > 1)
						tex.y -= 1;
					while (tex.y < 0)
						tex.y += 1;
					float tex_x = tex.x * tex_width;
					float tex_y = (1 - tex.y) * tex_height;
					int ppx = tex_x, ppy = tex_y;
					float wx = tex_x - ppx, wy = tex_y - ppy;
					if (ppx >= tex_width - 1)
						ppx -= 1;
					if (ppy >= tex_height - 1)
						ppy -= 1;
					unsigned char* rgb1 = tex_image + (ppy * tex_width + ppx) * 3;
					unsigned char* rgb2 = tex_image + (ppy * tex_width + ppx + 1) * 3;
					unsigned char* rgb3 = tex_image + (ppy * tex_width + ppx + tex_width) * 3;
					unsigned char* rgb4 = tex_image + (ppy * tex_width + ppx + tex_width + 1) * 3;
					unsigned char* output_rgb = color_image + pixel * 3;
					float intensity = 1;//0.3 + 0.7 * std::abs(glm::dot(light_dir, normal));
					for (int t = 0; t < 3; ++t) {
						output_rgb[t] = ((rgb1[t] * (1 - wx) + rgb2[t] * wx) * (1 - wy) + (rgb3[t] * (1 - wx) + rgb4[t] * wx) * wy) * intensity;
						//printf("%f ", (rgb1[t] * (1 - wx) + rgb2[t] * wx) * (1 - wy) + (rgb3[t] * (1 - wx) + rgb4[t] * wx) * wy);
					}
					//printf("%d\n", pixel);
					zbuffer[pixel] = z_quantize;
				}
			}
		}
	}}
};