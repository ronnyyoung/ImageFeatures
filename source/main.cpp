#include "imgFeat.h"

int main(int argc, char** argv)
{
	Mat image = imread(argv[1]);
	Mat edge;
	feat::getCannyEdge(image, edge);
	namedWindow("edge");
	imshow("edge", edge);
	waitKey();
	return 0;
}
