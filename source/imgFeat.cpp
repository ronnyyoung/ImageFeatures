#include "imgFeat.h"

void Feat::ExtBlobFeat(Mat& imgSrc, vector<SBlob> blobs)
{
	double dSigmaStart = 2;
	double dSigmaEnd = 15;
	double dSigmaStep = 1;

	Mat image;
	cvtColor(imgSrc, image, CV_BGR2GRAY);
	image.convertTo(image, CV_64F);

	vector<double> ivecSigmaArray;
	double dInitSigma = dSigmaStart;
	while (dInitSigma <= dSigmaEnd)
	{
		ivecSigmaArray.push_back(dInitSigma);
		dInitSigma += dSigmaStep;
	}
	int iSigmaNb = ivecSigmaArray.size();

	vector<Mat> matVecLOG;
	
	for (size_t i = 0; i != iSigmaNb; i++)
	{
		double iSigma = ivecSigmaArray[i]; 

		cout << iSigma << endl;
		
		Size kSize(6 * iSigma + 1, 6 * iSigma + 1);
		Mat HOGKernel = getHOGKernel(kSize, iSigma);
		Mat imgLog;
		
		filter2D(image, imgLog, -1, HOGKernel); // why imgLog must be an empty mat ?
		imgLog = imgLog * iSigma *iSigma;

		matVecLOG.push_back(imgLog);
	}

	vector<SBlob> allBlobs;
	for (size_t k = 1; k != matVecLOG.size() - 1 ;k++)
	{
		Mat topLev = matVecLOG[k + 1];
		Mat medLev = matVecLOG[k];
		Mat botLev = matVecLOG[k - 1];
		for (int i = 1; i < image.rows - 1; i++)
		{
			double* pTopLevPre = topLev.ptr<double>(i - 1);
			double* pTopLevCur = topLev.ptr<double>(i);
			double* pTopLevAft = topLev.ptr<double>(i + 1);

			double* pMedLevPre = medLev.ptr<double>(i - 1);
			double* pMedLevCur = medLev.ptr<double>(i);
			double* pMedLevAft = medLev.ptr<double>(i + 1);

			double* pBotLevPre = botLev.ptr<double>(i - 1);
			double* pBotLevCur = botLev.ptr<double>(i);
			double* pBotLevAft = botLev.ptr<double>(i + 1);

			for (int j = 1; j < image.cols - 1; j++)
			{
				if ((pMedLevCur[j] >= pMedLevCur[j + 1] && pMedLevCur[j] >= pMedLevCur[j -1] &&
				pMedLevCur[j] >= pMedLevPre[j + 1] && pMedLevCur[j] >= pMedLevPre[j -1] && pMedLevCur[j] >= pMedLevPre[j] && 
				pMedLevCur[j] >= pMedLevAft[j + 1] && pMedLevCur[j] >= pMedLevAft[j -1] && pMedLevCur[j] >= pMedLevAft[j] &&
				pMedLevCur[j] >= pTopLevPre[j + 1] && pMedLevCur[j] >= pTopLevPre[j -1] && pMedLevCur[j] >= pTopLevPre[j] &&
				pMedLevCur[j] >= pTopLevCur[j + 1] && pMedLevCur[j] >= pTopLevCur[j -1] && pMedLevCur[j] >= pTopLevCur[j] &&
				pMedLevCur[j] >= pTopLevAft[j + 1] && pMedLevCur[j] >= pTopLevAft[j -1] && pMedLevCur[j] >= pTopLevAft[j] &&
				pMedLevCur[j] >= pBotLevPre[j + 1] && pMedLevCur[j] >= pBotLevPre[j -1] && pMedLevCur[j] >= pBotLevPre[j] &&
				pMedLevCur[j] >= pBotLevCur[j + 1] && pMedLevCur[j] >= pBotLevCur[j -1] && pMedLevCur[j] >= pBotLevCur[j] &&
				pMedLevCur[j] >= pBotLevAft[j + 1] && pMedLevCur[j] >= pBotLevAft[j -1] && pMedLevCur[j] >= pBotLevAft[j] ) || 
				(pMedLevCur[j] < pMedLevCur[j + 1] && pMedLevCur[j] < pMedLevCur[j -1] &&
				pMedLevCur[j] < pMedLevPre[j + 1] && pMedLevCur[j] < pMedLevPre[j -1] && pMedLevCur[j] < pMedLevPre[j] && 
				pMedLevCur[j] < pMedLevAft[j + 1] && pMedLevCur[j] < pMedLevAft[j -1] && pMedLevCur[j] < pMedLevAft[j] &&
				pMedLevCur[j] < pTopLevPre[j + 1] && pMedLevCur[j] < pTopLevPre[j -1] && pMedLevCur[j] < pTopLevPre[j] &&
				pMedLevCur[j] < pTopLevCur[j + 1] && pMedLevCur[j] < pTopLevCur[j -1] && pMedLevCur[j] < pTopLevCur[j] &&
				pMedLevCur[j] < pTopLevAft[j + 1] && pMedLevCur[j] < pTopLevAft[j -1] && pMedLevCur[j] < pTopLevAft[j] &&
				pMedLevCur[j] < pBotLevPre[j + 1] && pMedLevCur[j] < pBotLevPre[j -1] && pMedLevCur[j] < pBotLevPre[j] &&
				pMedLevCur[j] < pBotLevCur[j + 1] && pMedLevCur[j] < pBotLevCur[j -1] && pMedLevCur[j] < pBotLevCur[j] &&
				pMedLevCur[j] < pBotLevAft[j + 1] && pMedLevCur[j] < pBotLevAft[j -1] && pMedLevCur[j] < pBotLevAft[j] ))
				{
					SBlob blob;
					blob.position = Point(j, i);
					blob.sigma = ivecSigmaArray[k];
					blob.value = pMedLevCur[j];
					allBlobs.push_back(blob);
				}
			}
		}
	}

	

	vector<bool> delFlags(allBlobs.size(), true);
	for (size_t i = 0; i != allBlobs.size(); i++)
	{
		if (delFlags[i] == false)
		{
			continue;
		}
		for (size_t j = i; j != allBlobs.size(); j++)
		{
			if (delFlags[j] == false)
			{
				continue;
			}
			double distCent = sqrt((allBlobs[i].position.x - allBlobs[j].position.x) * (allBlobs[i].position.x - allBlobs[j].position.x) + 
			(allBlobs[i].position.y - allBlobs[j].position.y) * (allBlobs[i].position.y - allBlobs[j].position.y));
			if ((allBlobs[i].sigma + allBlobs[j].sigma) / distCent > 2)
			{
				if (allBlobs[i].value >= allBlobs[j].value)
				{
					delFlags[j] = false;
					delFlags[i] = true;
				}
				else
				{
				 	delFlags[i] = false;
				 	delFlags[j] = true;
				}
			}
		}
	}


	for (size_t i = 0; i != allBlobs.size(); i++)
	{
		if (delFlags[i])
		{
			blobs.push_back(allBlobs[i]);
		}
	}

	sort(blobs.begin(), blobs.end(), compareBlob);
	
}


Mat Feat::getHOGKernel(Size& ksize, double sigma)
{
	Mat kernel(ksize, CV_64F);
	Point centPoint = Point((ksize.width -1)/2, ((ksize.height -1)/2));
	// first calculate Gaussian
	for (int i=0; i < kernel.rows; i++)
	{
		double* pData = kernel.ptr<double>(i);
		for (int j = 0; j < kernel.cols; j++)
		{
			double param = -((i - centPoint.y) * (i - centPoint.y) + (j - centPoint.x) * (j - centPoint.x)) / (2*sigma*sigma);
			pData[j] = exp(param);
		}
	}
	double maxValue;
	minMaxLoc(kernel, NULL, &maxValue);
	for (int i=0; i < kernel.rows; i++)
	{
		double* pData = kernel.ptr<double>(i);
		for (int j = 0; j < kernel.cols; j++)
		{
			if (pData[j] < EPS* maxValue)
			{
				pData[j] = 0;
			}
		}
	}

	double sumKernel = sum(kernel)[0];
	if (sumKernel != 0)
	{
		kernel = kernel / sumKernel;
	}
	// now calculate Laplacian
	for (int i=0; i < kernel.rows; i++)
	{
		double* pData = kernel.ptr<double>(i);
		for (int j = 0; j < kernel.cols; j++)
		{
			double addition = ((i - centPoint.y) * (i - centPoint.y) + (j - centPoint.x) * (j - centPoint.x) - 2*sigma*sigma)/(sigma*sigma*sigma*sigma);
			pData[j] *= addition;
		}
	}
	// make the filter sum to zero
	sumKernel = sum(kernel)[0];
	kernel -= (sumKernel/(ksize.width  * ksize.height));	

	return kernel;
}

void getCannyEdge(const& Mat& imgSrc, double lowThresh = -1, double highThresh = -1, double sigma = 1)
{
	if (imgSrc.channels() == 3)
	{
		cvtColor(imgSrc, gray, CV_BGR2GRAY);
	}
	else
	{
		gray = imgSrc.clone();
	}
	gray.convertTo(gray, CV_64F);
	
	double gaussianDieOff = .0001;
	double percentOfPixelsNotEdges = .7; // Used for selecting thresholds
	double thresholdRatio = .4;   // Low thresh is this fraction of the high

	int possibleWidth = 30;
	double ssq = sigma * sigma;
	for (int i = 1; i <= initWidth; i++)
	{
		if (exp(-(i * i) / (2* ssq)) < gaussianDieOff)
		{
			possibleWidth = i - 1;
			break;
		}
	}
cout << possibleWidth << endl;
	if (possibleWidth == 30)
	{
		possibleWidth = 1; // the user entered a reallly small sigma
	}

	// get the 1D gaussian filter
	int winSz = 2 * possibleWidth + 1;
	Mat gaussKernel1D(1, windSz, CV_64F);
	double kernelPtr = gaussKernel1D.ptr<double>(0);
	for (int i = 0; i < gaussKernel1D.cols; i++)
	{
		kernelPtr[i] = exp((i - possibleWidth) * (i - possibleWidth) / (2 * ssq)) / (2 * CV_PI * ssq);
	}
	
	// get the derectional derivatives of gaussian kernel
	Mat dGaussKernel(1, windSz, CV_64F);
	double dKernelPtr = dGaussKernel.ptr<double>(0);
	for (int i = 0; i < dGaussKernel.cols; i++)
	{
		dKernelPtr[i] = - (i - possibleWidth) * exp((i - possibleWidth) * (i - possibleWidth) / (2 * ssq)) / (CV_PI * ssq);
	}

	/* smooth the image out*/
	Mat imgSmooth;
	filter2D(gray, imgSmooth, -1, gaussKernel1D);
	filter2D(imgSmooth, imgSmooth, -1, gaussKernel1D.t());
	/*apply directional derivatives*/

	Mat imgX, imgY;
	filter2D(imgSmooth, imgX, -1, dGaussKernel);
	filter2D(imgSmooth, imgY, -1, dGaussKernel.t());

	Mat imgMag;
	sqrt(imgX.mul(imgX) + imgY.mul(imgY), imgMag);
	double magMax;
	minMaxLoc(imgMag, NULL, &magMax, NULL, NULL);
	if (magMax > 0 )
	{
		imgMag = imgMag / magMax;
	}
	
	if (lowThresh == -1 || highThresh == -1)
	{
		
	}
	
}

double getGaussianThresh(const Mat& inputArray, double percentage)
{
	double thresh = -1;
	// compute the 64-hist of inputArray
	int nBins = 64;
	double minValue, maxVaule;
	minMaxLoc(inputArray, &minValue, &maxValue, NULL, NULL);
	double step = (maxValue - minValue) / nBins;
	vector<unsigned> histBin(nBins,0);
	for (int i = 0; i < inputArray.rows; i++)
	{
		double *pData = inputArray.ptr<double>(i);
		for(int j = 0; j < inputArray.cols; j++)
		{
			int index = (pData[j] - minValue) / step;
			histBin[index]++;
		}
	}
	unsigned cumSum = 0; 
	for (int i = 0; i < nBins; i++)
	{
		cumSum += histBin[i];
		if (cumSun > percentage * inputArray.rows * inputArray.cols)
		{
			thresh = (i + 1) / 64;
			break;
		}
	}
	return thresh;
	
}

