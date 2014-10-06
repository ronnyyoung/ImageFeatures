#include "imgFeat.h"

void feat::extBlobFeat(Mat& imgSrc, vector<SBlob>& blobs)
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


Mat feat::getHOGKernel(Size& ksize, double sigma)
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

bool feat::compareBlob(const SBlob& lhs, const SBlob& rhs)
{
	return lhs.value > rhs.value;
}


