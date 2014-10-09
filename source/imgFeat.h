#ifndef IMGFEAT_H_
#define IMGFEAT_H_

#include <iostream>
#include <algorithm>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

namespace feat
{
	const double EPS = 2.2204e-16;
	struct SBlob
	{
		Point position;
		double value;
		double sigma;
	};
	static bool compareBlob(const SBlob& lhs, const SBlob& rhs);
	Mat getHOGKernel(Size& ksize, double sigma);
	void extBlobFeat(Mat& imgSrc, vector<SBlob>& blobs);


	// edge detection
	enum sobelDirection{SOBEL_HORZ, SOBEL_VERT, SOBEL_BOTH};
	double getSobelEdge(const Mat& imgSrc, Mat& imgDst, double thresh = -1, int direction = SOBEL_BOTH);
	static double getCannyThresh(const Mat& inputArray, double percentage);
	void getCannyEdge(const Mat& imgSrc, Mat& imgDst, double lowThresh = -1, double highThresh = -1, double sigma = 1);


	void detectHarrisCorners(const Mat& imgSrc, Mat& imgDst, double alpha);
	void drawCornerOnImage(Mat& image, const Mat&binary);
	void detectHarrisLaplace(const Mat& imgSrc, Mat& imgDst);
}


#endif
