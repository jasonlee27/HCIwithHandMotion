#include <opencv/cv.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d\features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect\objdetect.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/video/tracking.hpp>

#include <cstdio>
#include <iostream>
#include <Windows.h>
#include <WinUser.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <math.h>
#include <sstream>
#include "stdafx.h"
#include "Header.h"

using namespace cv;
using namespace std;
void BackgroundImgSubtract(Mat &, Mat &, Mat &);
void skinDetection(Mat &, Mat &, vector<Rect>&);
void findHandContour(vector<vector<Point>>&, Mat&, vector<Vec4i>);
float euclideanDist(Point &, Point &);
int findPalmCircleRad(Mat, vector<Point>, Point);
Rect camshift_find(Mat &faceFrame, Rect facesPart, int trackFace);

void GestureClassification() {


	Mat white;
	Scalar lineColor = Scalar(255, 255, 255);
	Mat BgImg, DiffImg;
	Rect HandWindow;
	Mat FrameImg;
	Mat SkinImg; /* After filtering and thresholding */
	int lineSize;
	vector<vector<Point>> contours; /* Hand contour */
	vector<Point> contoursApp;
	vector<vector<int>> hullI;  /* Hand convex hull */
	vector<vector<Point>> hull;
	vector<vector<Vec4i>> cnvxDefects;
	vector<Rect> faces;
	vector<Moments> mu;
	vector<Point> mc, Pstmc; /* Current and Previous Hand center */
	vector<Point2f> enc_cnt; /* Enclosing circle center */
	vector<float> enc_r; /* Enclosing circle radius */
	vector<vector<Point>> fingers;  /* Detected fingers positions */
	CascadeClassifier face_cascade;
	VideoCapture cap;
	Ptr<BackgroundSubtractor> bgSubtractor = createBackgroundSubtractorMOG2(500, 16, true);
	int BGRselect = -1;/* 0: Blue, 1: Green, 2: Red, -1: No color selected */

	/* getting the size of laptop */
	RECT laptop;
	GetWindowRect(GetDesktopWindow(), &laptop);

	face_cascade.load("C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml");
	if (face_cascade.empty()) {
		cout << "face_cascade xml file is failed  " << face_cascade.empty() << endl;
	}
	else
		cout << "face_cascade xml file is loaded!!  " << face_cascade.empty() << endl;
	// Capturing background for bg subtraction **********************************************************************************************
	cap.open(0);
	int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	white = Mat(height, width, CV_8UC3, Scalar(255, 255, 255));

	while (waitKey(1) != 32) {
		cap >> FrameImg;
		if (FrameImg.empty())
			break; // End of video stream
		flip(FrameImg, FrameImg, 1);
		imshow("Video Recording now. Press SPACE for taking a picture of background", FrameImg);
	}
	cap.retrieve(BgImg);
	flip(BgImg, BgImg, 1);
	destroyAllWindows();

	// MAIN LOOP *****************************************************************************************************************************
	while (waitKey(30) != 32) {
		cap >> FrameImg;
		if (FrameImg.empty())
			break; // End of video stream
		flip(FrameImg, FrameImg, 1);

		/* Background subtraction */
		Mat DiffImgCh[3], FrameImgCh[3], BgImgCh[3];
		Mat FrameYCbCr, BgYCbCr, MaskImg;
		int thresh[3] = { 100, 100, 100 };
		cvtColor(FrameImg, FrameYCbCr, CV_BGR2YCrCb);
		cvtColor(BgImg, BgYCbCr, CV_BGR2YCrCb);
		split(FrameYCbCr, FrameImgCh);
		split(BgYCbCr, BgImgCh);

		/* 0:Y, 1:Cb, 2:Cr */
		for (int i = 0; i < 3; i++)  {
			absdiff(FrameImgCh[i], BgImgCh[i], DiffImgCh[i]);
			threshold(DiffImgCh[i], DiffImgCh[i], thresh[i], 255, THRESH_BINARY + THRESH_OTSU);
		}
		DiffImg = DiffImgCh[0] | DiffImgCh[1] | DiffImgCh[2];
		erode(DiffImg, DiffImg, Mat());
		GaussianBlur(DiffImg, DiffImg, Size(9, 9), 0);
		FrameImg.copyTo(MaskImg, DiffImg);
		imshow("Background", MaskImg);

		/* find face using haar-cascade */
		if (!face_cascade.empty()) {
			face_cascade.detectMultiScale(FrameImg, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		}

		/* skin detection and find skin contours */
		vector<Vec4i> hierarchy;
		skinDetection(MaskImg, SkinImg, faces); //Skin Image contains the detected skin
		vector<Vec3f> circles;
		HoughCircles(SkinImg, circles, CV_HOUGH_GRADIENT, 1, SkinImg.rows / 8, 200, 11, 10, 30);

		for (size_t i = 0; i < circles.size(); i++)
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			//circle(FrameImg, center, 3, Scalar(0, 255, 0), -1, 8, 0);
			// circle outline
			circle(FrameImg, center, radius, Scalar(0, 0, 255), 3, 8, 0);
		}
		findHandContour(contours, SkinImg.clone(), hierarchy); //contours is the vector of contour(vector of points)

		mu = vector<Moments>(contours.size());// moments; used for centroid of the centroid.
		mc = vector<Point>(contours.size());  // center/oid of contour 
		if (!Pstmc.size())  {
			Pstmc = vector<Point>(contours.size());
		} // past centroid
		enc_cnt = vector<Point2f>(contours.size()); // centr of enclosing circle.
		cnvxDefects = vector<vector<Vec4i>>(contours.size()); //  
		hullI = vector<vector<int>>(contours.size()); // 
		enc_r = vector<float>(contours.size()); // radius of enclosing circle 
		Point min_point;
		int numless90, numConvex;
		int rPalm;
		int mindist = FrameImg.size().height;

		for (int i = 0; i< contours.size(); i++) {
			if (contours.size()) {

				/* draw contours and find minimum enclosing circle */
				vector<float> pointAngle = vector<float>(contours[i].size());
				vector<int> distVec = vector<int>(contours[i].size());
				mu[i] = moments(contours[i]);
				mc[i] = Point(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
				minEnclosingCircle(contours[i], enc_cnt[i], enc_r[i]);
				int radiusIn = findPalmCircleRad(MaskImg, contours[i], mc[i]);
				circle(FrameImg, enc_cnt[i], enc_r[i], Scalar(159, 0, 255), 2, 8, 0);
				circle(FrameImg, mc[i], radiusIn, Scalar(0, 50, 0), 2, 8, 0);

				approxPolyDP(Mat(contours[i]), contours[i], arcLength(Mat(contours[i]), true)*0.032, true);
				drawContours(FrameImg, contours, i, Scalar(0, 255, 0), 2, 8, hierarchy);

				/* find convex hull and defects */
				float Dist; float maxDist = 0;
				convexHull(Mat(contours[i]), hullI[i], false);
				convexityDefects(Mat(contours[i]), hullI[i], cnvxDefects[i]);
				numConvex = 0;

				if (cnvxDefects[i].size())  {
					for (vector<Vec4i> ::iterator it = cnvxDefects[i].begin(); it != cnvxDefects[i].end();) {
						Vec4i& v = (*it);
						int start_indx = v[0];
						int end_indx = v[1];
						int far_indx = v[2];
						int depth = v[3] / 256;
						float a, b, c;
						int angle;
						Point ptStart = (contours[i][start_indx]);
						Point ptEnd = (contours[i][end_indx]);//point of the contour where the defect ends
						Point ptFar = (contours[i][far_indx]);//the farthest from the convex point within the defect
						a = sqrt((ptEnd.x - ptStart.x)*(ptEnd.x - ptStart.x) + (ptEnd.y - ptStart.y)*(ptEnd.y - ptStart.y));
						b = sqrt((ptFar.x - ptStart.x)*(ptFar.x - ptStart.x) + (ptFar.y - ptStart.y)*(ptFar.y - ptStart.y));
						c = sqrt((ptEnd.x - ptFar.x)*(ptEnd.x - ptFar.x) + (ptEnd.y - ptFar.y)*(ptEnd.y - ptFar.y));

						angle = acos((b*b + c*c - a*a) / (2 * b*c)) * 57;
						if (angle <= 90 && depth>radiusIn && depth<enc_r[i] && contours[i][far_indx].y <= mc[i].y + 10)   {
							numConvex++;
							circle(FrameImg, ptFar, 5, Scalar(0, 0, 255), -1);
							circle(FrameImg, ptStart, 5, Scalar(255, 255, 255), -1);
							//line(FrameImg, mc[i], ptStart, Scalar(0, 0, 0), 2, 8, 0);
						}
						++it;
					}
				}

				/* find palm circle of  */
				if (!contours[i].size())
					break;
				else {
					numless90 = 0;
					for (int j = 0; j < contours[i].size(); j = j + 1)   {
						int dist = sqrt((mc[i].x - contours[i][j].x) * (mc[i].x - contours[i][j].x) + (mc[i].y - contours[i][j].y) * (mc[i].y - contours[i][j].y));
						distVec[j] = dist;
						if (dist <= mindist && dist > 0)  {
							mindist = dist;
						}
						min_point = contours[i][j];

						Point midPnt = contours[i][j];
						Point leftPnt = contours[i][j + 1];
						Point rightPnt = contours[i][j - 1];
						if (j == 0) {
							rightPnt = contours[i][contours[i].size() - 1];
						}
						else if (j == contours[i].size() - 1)   {
							leftPnt = contours[i][1];
						}

						/* calculate degree for each point */
						Point mid2leftVec = leftPnt - midPnt;
						Point mid2rightVec = rightPnt - midPnt;
						float dot = mid2leftVec.x*mid2rightVec.x + mid2leftVec.y*mid2rightVec.y;
						float magmult = sqrt((mid2leftVec.x*mid2leftVec.x + mid2leftVec.y*mid2leftVec.y)*(mid2rightVec.x*mid2rightVec.x + mid2rightVec.y*mid2rightVec.y));
						pointAngle[j] = acos(dot / magmult) * 180 / 3.14159265;
						if (pointAngle[j] > 0 && pointAngle[j] < 60 && dist >= 1.5*radiusIn && midPnt.y <= mc[i].y + 10)  {
							line(FrameImg, mc[i], midPnt, Scalar(0, 0, 0), 2, 8, 0);
							//line(FrameImg, midPnt, leftPnt, Scalar(0, 255, 0), 2, 8, 0);
							//circle(FrameImg, midPnt, 8, Scalar(255, 0, 0), 5);
							numless90++;
						}
					}

				}

				//putText(FrameImg, to_string(numless90), Point(50, 100), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(200, 0, 200), 2, CV_AA);
				//putText(FrameImg, to_string(numConvex), Point(50, 400), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);
				rPalm = mindist + 40;
				//circle(FrameImg, mc[i], rPalm, Scalar(255, 0, 0), 2, 8, 0);
				float diffR = enc_r[i] / mindist;
				//putText(FrameImg, to_string(diffR), Point(50, 300), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);

				//putText(FrameImg, to_string(numConvex), Point(50, 150), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 0, 0), 2, CV_AA);
				//putText(FrameImg, to_string(diffR), Point(50, 200), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 2, CV_AA);
				if (circles.size() == 0 && diffR<1.6 && (numless90 == 0) && (numConvex == 0)) {
					putText(FrameImg, "CLOSE PALM", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);
				}
				else if (circles.size() == 0 && (diffR>1.8 && diffR<2.9) && (numless90 <= 3) && (numConvex == 0)) {
					putText(FrameImg, "OPEN PALM", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);
				}
				else if (circles.size()>0 && (numless90 == 1 || numless90 == 2)){
					putText(FrameImg, "OKAY sign", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);
				}
				if ((numless90 == 1 && numConvex == 0))  {
					putText(FrameImg, "ONE", Point(400, 50), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);
				}
				else if (numless90 == 2 && numConvex <= 2)   {
					putText(FrameImg, "TWO", Point(400, 50), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);
				}
				else if (numless90 == 3 && numConvex <= 2)    {
					putText(FrameImg, "THREE", Point(400, 50), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);
				}
				else if (numless90 == 4)   {
					putText(FrameImg, "FOUR", Point(400, 50), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);
				}
				else if (numless90 == 5)    {
					putText(FrameImg, "FIVE", Point(400, 50), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);
				}

				Pstmc[i] = mc[i];
			}
		}

		imshow("Skin", SkinImg);
		imshow("Original with Contours", FrameImg);
	}
	cap.~VideoCapture();
	destroyAllWindows();
}
