#include "imgFeat.h"

int main(int argc, char** argv)
{
	Mat image = imread(argv[1]);
	Mat cornerMap;
	feat::detectHarrisLaplace(image ,cornerMap);
	feat::drawCornerOnImage(image, cornerMap);
	
	namedWindow("corners");
	imshow("corners", image);
	waitKey();
	return 0;
}
