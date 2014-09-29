#include "imgFeat.h"

int main(int argc, char** argv)
{
	Mat image = imread(argv[1]);
	Feat::ExtBlobFeat(image);
	return 0;
}
