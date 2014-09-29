#include "imgFeat.h"

int main(int argc, char** argv)
{
	Mat image = imread(argv[1]);
	vector<Feat::SBlob> blobs;
	Feat::ExtBlobFeat(image, blobs);
	return 0;
}
