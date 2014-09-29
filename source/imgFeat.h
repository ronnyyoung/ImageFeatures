#ifndef IMGFEAT_H_
#define IMGFEAT_H_

#include <iostream>
#include <algorithm>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

namespace Feat
{
	const double EPS = 2.2204e-16;
	struct SBlob
	{
		Point position;
		double value;
		double sigma;
	};
	inline bool compareBlob(const SBlob& lhs, const SBlob& rhs)
	{
		return lhs.value > rhs.value;
	}
	Mat getHOGKernel(Size& ksize, double sigma);
	void ExtBlobFeat(Mat& image);
}


#endif
