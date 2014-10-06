#include "imgFeat.h"

/*
* @function: specifies the canny method.

* @param IMGSRC: the input image
* @param IMGDST: canny edge image
* @param LOWTHRESH: the low threshold
* @param HIGHTHRESH: the high threshold
* @param SIGMA: SIGMA is the standard deviation of the Gaussian filter. The default SIGMA is 1.
* the size of the filter is chosen automatically , based on SIGMA

* Revision:1.o
* Data:2014/10/6
* Athor:Ronny
*/

void feat::getCannyEdge(const Mat& imgSrc, Mat& imgDst, double lowThresh, double highThresh, double sigma)
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
    gray = gray / 255;
    
    double gaussianDieOff = .0001;
    double percentOfPixelsNotEdges = .7; // Used for selecting thresholds
    double thresholdRatio = .4;   // Low thresh is this fraction of the high

    int possibleWidth = 30;
    double ssq = sigma * sigma;
    for (int i = 1; i <= possibleWidth; i++)
    {
        if (exp(-(i * i) / (2* ssq)) < gaussianDieOff)
        {
            possibleWidth = i - 1;
            break;
        }
    }

    if (possibleWidth == 30)
    {
        possibleWidth = 1; // the user entered a reallly small sigma
    }

    // get the 1D gaussian filter
    int winSz = 2 * possibleWidth + 1;
    Mat gaussKernel1D(1, winSz, CV_64F);
    double* kernelPtr = gaussKernel1D.ptr<double>(0);
    for (int i = 0; i < gaussKernel1D.cols; i++)
    {
        kernelPtr[i] = exp(-(i - possibleWidth) * (i - possibleWidth) / (2 * ssq)) / (2 * CV_PI * ssq);
    }

    
    // get the derectional derivatives of gaussian kernel
    Mat dGaussKernel(winSz, winSz, CV_64F);
    for (int i = 0; i < dGaussKernel.rows; i++)
    {
        double* linePtr = dGaussKernel.ptr<double>(i);
        for (int j = 0; j< dGaussKernel.cols; j++)
        {
            linePtr[j] = - (j - possibleWidth) * exp(-((i - possibleWidth) * (i - possibleWidth) + (j - possibleWidth) * (j - possibleWidth)) / (2 * ssq)) / (CV_PI * ssq);
        }
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
        highThresh = getCannyThresh(imgMag, percentOfPixelsNotEdges);
        lowThresh = thresholdRatio * highThresh;
    }




    Mat imgStrong = Mat::zeros(imgMag.size(), CV_8U);
    Mat imgWeak = Mat::zeros(imgMag.size(), CV_8U);
    
    
    for (int dir = 1; dir <= 4; dir++)
    {
        Mat gradMag1(imgMag.size(), imgMag.type());
        Mat gradMag2(imgMag.size(), imgMag.type());
        Mat idx = Mat::zeros(imgMag.size(), CV_8U);
        if (dir == 1)
        {
            Mat dCof = abs(imgY / imgX);
            idx = (imgY <= 0 & imgX > -imgY) | (imgY >= 0 & imgX < -imgY);
            idx.row(0).setTo(Scalar(0));
            idx.row(idx.rows - 1).setTo(Scalar(0));
            idx.col(0).setTo(Scalar(0));
            idx.col(idx.cols - 1).setTo(Scalar(0));
            for (int i = 1; i < imgMag.rows - 1; i++)
            {
                for (int j = 1; j < imgMag.cols - 1; j++)
                {
                    gradMag1.at<double>(i,j) = (1 - dCof.at<double>(i,j)) * imgMag.at<double>(i,j + 1) + dCof.at<double>(i,j) * imgMag.at<double>(i - 1,j + 1);
                    gradMag2.at<double>(i,j) = (1 - dCof.at<double>(i,j)) * imgMag.at<double>(i,j - 1) + dCof.at<double>(i,j) * imgMag.at<double>(i + 1,j - 1);
                }
            }
        }
        else if(dir == 2)
        {
            Mat dCof = abs(imgX / imgY);
            idx = (imgX > 0 & -imgY >= imgX) | (imgX < 0 & -imgY <= imgX);
            for (int i = 1; i < imgMag.rows - 1; i++)
            {
                for (int j = 1; j < imgMag.cols - 1; j++)
                {
                    gradMag1.at<double>(i,j) = (1 - dCof.at<double>(i,j)) * imgMag.at<double>(i - 1,j) + dCof.at<double>(i,j) * imgMag.at<double>(i - 1,j + 1);
                    gradMag2.at<double>(i,j) = (1 - dCof.at<double>(i,j)) * imgMag.at<double>(i + 1,j) + dCof.at<double>(i,j) * imgMag.at<double>(i + 1,j - 1);
                }
            }
        }
        else if(dir == 3)
        {
            Mat dCof = abs(imgX / imgY);
            idx = (imgX <= 0 & imgX > imgY) | (imgX >= 0 & imgX < imgY);
            for (int i = 1; i < imgMag.rows - 1; i++)
            {
                for (int j = 1; j < imgMag.cols - 1; j++)
                {
                    gradMag1.at<double>(i,j) = (1 - dCof.at<double>(i,j)) * imgMag.at<double>(i - 1,j) + dCof.at<double>(i,j) * imgMag.at<double>(i - 1,j - 1);
                    gradMag2.at<double>(i,j) = (1 - dCof.at<double>(i,j)) * imgMag.at<double>(i + 1,j) + dCof.at<double>(i,j) * imgMag.at<double>(i + 1,j + 1);
                }
            }
        
        }
        else
        {
            Mat dCof = abs(imgY / imgX);
            idx = (imgY <0 & imgX <= imgY) | (imgY > 0 & imgX >= imgY);
            for (int i = 1; i < imgMag.rows - 1; i++)
            {
                for (int j = 1; j < imgMag.cols - 1; j++)
                {
                    gradMag1.at<double>(i,j) = (1 - dCof.at<double>(i,j)) * imgMag.at<double>(i,j - 1) + dCof.at<double>(i,j) * imgMag.at<double>(i - 1,j - 1);
                    gradMag2.at<double>(i,j) = (1 - dCof.at<double>(i,j)) * imgMag.at<double>(i,j + 1) + dCof.at<double>(i,j) * imgMag.at<double>(i + 1,j + 1);
                }
            }
        }

        Mat idxLocalMax = idx & ((imgMag >= gradMag1) & (imgMag >= gradMag2));


        imgWeak = imgWeak | ((imgMag > lowThresh) & idxLocalMax);
        imgStrong= imgStrong| ((imgMag > highThresh) & imgWeak);

    }

    imgDst = Mat::zeros(imgWeak.size(),imgWeak.type());
    for (int i = 1; i < imgWeak.rows - 1; i++)
    {
        uchar* pWeak = imgWeak.ptr<uchar>(i);
        uchar* pDst = imgDst.ptr<uchar>(i);
        uchar* pStrPre = imgStrong.ptr<uchar>(i - 1);
        uchar* pStrMid = imgStrong.ptr<uchar>(i);
        uchar* pStrAft = imgStrong.ptr<uchar>(i + 1);
        for (int j = 1; j < imgWeak.cols - 1; j++)
        {
            if (!pWeak[j])
            {
                continue;
            }
            if (pStrMid[j])
            {
                pDst[j] = 255;
            }
            if (pStrMid[j-1] || pStrMid[j+1] || pStrPre[j-1] || pStrPre[j] || pStrPre[j+1] || pStrAft[j-1] || pStrAft[j] ||pStrAft[j+1])
            {
                pDst[j] = 255;
            }
        }
    }
}

double feat::getCannyThresh(const Mat& inputArray, double percentage)
{
    double thresh = -1.0;
    // compute the 64-hist of inputArray
    int nBins = 64;
    double minValue, maxValue;
    minMaxLoc(inputArray, &minValue, &maxValue, NULL, NULL);
    double step = (maxValue - minValue) / nBins;

    vector<unsigned> histBin(nBins,0);
    for (int i = 0; i < inputArray.rows; i++)
    {
        const double* pData = inputArray.ptr<double>(i);
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

        if (cumSum > percentage * inputArray.rows * inputArray.cols)
        {
            thresh = (i + 1) / 64.0;
            break;
        }
    }
    return thresh;
    
}

