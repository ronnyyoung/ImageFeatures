#include "imgFeat.h"

void feat::detectHarrisLaplace(const Mat& imgSrc, Mat& imgDst)
{
	Mat gray;
	if (imgSrc.channels() == 3)
	{
		cvtColor(imgSrc, gray, CV_BGR2GRAY);
	}
	else
	{
		gray = imgSrc.clone();
	}
	gray.convertTo(gray, CV_64F);

	/* 尺度设置*/
	double dSigmaStart = 1.5;
	double dSigmaStep = 1.2;
	int iSigmaNb = 13;

	vector<double> dvecSigma(iSigmaNb);
	for (int i = 0; i < iSigmaNb; i++)
	{
		dvecSigma[i] = dSigmaStart + i*dSigmaStep;
	}
	vector<Mat> harrisArray(iSigmaNb);
	
	for (int i = 0; i < iSigmaNb; i++)
	{
		double iSigmaI = dvecSigma[i];
		double iSigmaD = 0.7 * iSigmaI;

		int iKernelSize = 6*round(iSigmaD) + 1;
		/*微分算子*/
		Mat dx(1, iKernelSize, CV_64F);
		for (int k =0; k < iKernelSize; k++)
		{
			int pCent = (iKernelSize - 1) / 2;
			int x = k - pCent;
			dx.at<double>(0,i) = x * exp(-x*x/(2*iSigmaD*iSigmaD))/(iSigmaD*iSigmaD*iSigmaD*sqrt(2*CV_PI));
		}
	
		Mat dy = dx.t();
		Mat Ix,Iy;
		/*图像微分*/
		filter2D(gray, Ix, CV_64F, dx);
		filter2D(gray, Iy, CV_64F, dy);

		Mat Ix2,Iy2,Ixy;
		Ix2 = Ix.mul(Ix);
		Iy2 = Iy.mul(Iy);
		Ixy = Ix.mul(Iy);

		int gSize = 6*round(iSigmaI) + 1;
		Mat gaussKernel = getGaussianKernel(gSize, iSigmaI);
		filter2D(Ix2, Ix2, CV_64F, gaussKernel);
		filter2D(Iy2, Iy2, CV_64F, gaussKernel);
		filter2D(Ixy, Ixy, CV_64F, gaussKernel);

		/*自相关矩阵*/
		double alpha = 0.06;
		Mat detM = Ix2.mul(Iy2) - Ixy.mul(Ixy);
		Mat trace = Ix2 + Iy2;
		Mat cornerStrength = detM - alpha * trace.mul(trace);

		

		double maxStrength;
		minMaxLoc(cornerStrength, NULL, &maxStrength, NULL, NULL);
		Mat dilated;
		Mat localMax;
		dilate(cornerStrength, dilated, Mat());
		compare(cornerStrength, dilated, localMax, CMP_EQ);
	

		Mat cornerMap;
		double qualityLevel = 0.2;
		double thresh = qualityLevel * maxStrength;
		cornerMap = cornerStrength > thresh;
		bitwise_and(cornerMap, localMax, cornerMap);
		harrisArray[i] = cornerMap.clone();	
	}

	/*计算尺度归一化Laplace算子*/
	vector<Mat> laplaceSnlo(iSigmaNb);
	for (int i = 0; i < iSigmaNb; i++)
	{
		double iSigmaL = dvecSigma[i];
		Size kSize = Size(6 * floor(iSigmaL) +1, 6 * floor(iSigmaL) +1);
		Mat hogKernel = getHOGKernel(kSize,iSigmaL);
		filter2D(gray, laplaceSnlo[i], CV_64F, hogKernel);
		laplaceSnlo[i] *= (iSigmaL * iSigmaL);
	}
	
	/*检测每个特征点在某一尺度LOG相应是否达到最大*/
	Mat corners(gray.size(), CV_8U, Scalar(0));
	for (int i = 0; i < iSigmaNb; i++)
	{
		for (int r = 0; r < gray.rows; r++)
		{
			for (int c = 0; c < gray.cols; c++)
			{
				if (i ==0)
				{
					if (harrisArray[i].at<uchar>(r,c) > 0 && laplaceSnlo[i].at<double>(r,c) > laplaceSnlo[i + 1].at<double>(r,c))
					{
						corners.at<uchar>(r,c) = 255;
					}
				}
				else if(i == iSigmaNb -1)
				{
					if (harrisArray[i].at<uchar>(r,c) > 0 && laplaceSnlo[i].at<double>(r,c) > laplaceSnlo[i - 1].at<double>(r,c))
					{
						corners.at<uchar>(r,c) = 255;
					}
				}
				else
				{
					if (harrisArray[i].at<uchar>(r,c) > 0 &&
					laplaceSnlo[i].at<double>(r,c) > laplaceSnlo[i + 1].at<double>(r,c) &&
					laplaceSnlo[i].at<double>(r,c) > laplaceSnlo[i - 1].at<double>(r,c))
					{
						corners.at<uchar>(r,c) = 255;
					}
				}
			}
		}
	}
	imgDst = corners.clone();
	
}

