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
	struct SBlob
	{
		Point position;
		double value;
		double sigma;
	};
	const double EPS = 2.2204e-16;
	Mat getHOGKernel(Size& ksize, double sigma);
	
	inline bool compareBlob(const SBlob& lhs, const SBlob& rhs)
	{
		return lhs.value > rhs.value;
	}
	void ExtBlobFeat(Mat& image);
}


#endif
