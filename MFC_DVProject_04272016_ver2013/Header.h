#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
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
#include <ctype.h>
#include <ctime>
#include "stdafx.h"

using namespace cv;
using namespace std;

void BackgroundImgSubtract(Mat &, Mat &, Mat &);
void skinDetection(Mat &, Mat &, vector<Rect>&);
void findHandContour(vector<vector<Point>>&, Mat&, vector<Vec4i>);
int findPalmCircleRad(Mat, vector<Point>, Point);
void DrawingApplication();
void HandGestureRecognition();
void GestureClassification();
void PacmanGame();


void DrawingApplication() {
	
	Scalar lineColor = Scalar(255, 255, 255);
	Mat BgImg, DiffImg;
	Rect HandWindow;
	Mat FrameImg;
	Mat SkinImg, WhiteBoard; /* After filtering and thresholding */
	int lineSize = 1;
	vector<vector<Point>> contours; /* Hand contour */
	vector<Point> contoursApp;
	vector<vector<int>> hullI;  /* Hand convex hull */
	vector<vector<Point>> hull;
	vector<vector<Vec4i>> cnvxDefects;
	vector<Rect> faces;
	vector<Moments> mu;
	vector<Point> mc;
	Point Pstmc; /* Current and Previous Hand center */
	vector<Point2f> enc_cnt; /* Enclosing circle center */
	vector<float> enc_r; /* Enclosing circle radius */
	vector<vector<Point>> fingers;  /* Detected fingers positions */
	CascadeClassifier face_cascade;
	VideoCapture cap;
	double minY = -1, minCr = -1, minCb = -1, maxY = -1, maxCr = -1, maxCb = -1;
	int BGRselect = -1;/* 0: Blue, 1: Green, 2: Red, -1: No color selected */

	void onMouse(int event, int x, int y, int /*flags*/, void* /*param*/);

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
	WhiteBoard = Mat(height, width, CV_8UC3, Scalar(255, 255, 255));

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

	// Selecting the mode for controling the computer *******************************************************************************************
	while (waitKey(1) != 32) {
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

		for (int i = 0; i < 3; i++)  {/* 0:Y, 1:Cb, 2:Cr */
			absdiff(FrameImgCh[i], BgImgCh[i], DiffImgCh[i]);
			threshold(DiffImgCh[i], DiffImgCh[i], thresh[i], 255, THRESH_BINARY + THRESH_OTSU);
		}
		DiffImg = DiffImgCh[0] | DiffImgCh[1] | DiffImgCh[2];
		erode(DiffImg, DiffImg, Mat());
		GaussianBlur(DiffImg, DiffImg, Size(9, 9), 0);
		FrameImg.copyTo(MaskImg, DiffImg);

		/* find face using haar-cascade */
		if (!face_cascade.empty()) {
			face_cascade.detectMultiScale(FrameImg, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		}

		/* skin detection and find skin contours */
		vector<Vec4i> hierarchy;
		skinDetection(MaskImg, SkinImg, faces); //Skin Image contains the detected skin

		///// DISTANCE TRANSFORM*******************************************************************************
		//Mat dist;
		//distanceTransform(SkinImg, dist, CV_DIST_L2, 3);
		//double minVal, maxVal;
		//Point minLoc, maxLoc;
		//minMaxLoc(dist, &minVal, &maxVal, &minLoc, &maxLoc);
		//circle(FrameImg, maxLoc, 3, Scalar(0, 0, 255), 2, 8, 0);
		///// END OF DISTANCE TRANSFORM

		////Then define your mask image
		//cv::Mat mask = cv::Mat::zeros(SkinImg.size(), SkinImg.type());

		////I assume you want to draw the circle at the center of your image, with a radius of 50
		//rectangle(mask, Point(maxLoc.x - 5 * maxVal, maxLoc.y - 5 * maxVal), Point(maxLoc.x + 5 * maxVal, maxLoc.y + 1 * maxVal), Scalar(255, 255, 255), -1, 8, 0);
		////circle(mask, maxLoc, 3*maxVal, cv::Scalar(255, 255,255), -1, 8, 0);
		//bitwise_and(SkinImg, mask, SkinImg);
		//imshow("mask", SkinImg);


		findHandContour(contours, SkinImg.clone(), hierarchy); //contours is the vector of contour(vector of points)

		mu = vector<Moments>(contours.size());// moments; used for centroid of the centroid.
		mc = vector<Point>(contours.size());  // center/oid of contour 
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
				vector<Point> modeSelection;

				/* draw contours and find minimum enclosing circle */
				mu[i] = moments(contours[i]);
				mc[i] = Point(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
				minEnclosingCircle(contours[i], enc_cnt[i], enc_r[i]);
				int radiusIn = findPalmCircleRad(MaskImg, contours[i], mc[i]);

				approxPolyDP(Mat(contours[i]), contours[i], arcLength(Mat(contours[i]), true)*0.04, true);
				drawContours(FrameImg, contours, i, Scalar(0, 255, 0), 2, 8, hierarchy);
				vector<float> pointAngle = vector<float>(contours[i].size());
				vector<int> distVec = vector<int>(contours[i].size());

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
						if (depth > radiusIn && depth < enc_r[i] && angle <= 90)   {
							numConvex++;
							circle(FrameImg, ptFar, 5, Scalar(0, 0, 255), -1);
							circle(FrameImg, ptStart, 5, Scalar(255, 255, 255), -1);
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
							circle(FrameImg, midPnt, 8, Scalar(255, 0, 0), 5);
							modeSelection.push_back(midPnt);
							numless90++;
						}
					}
					putText(FrameImg, to_string(numless90), Point(50, 100), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);
					putText(FrameImg, to_string(radiusIn), Point(50, 400), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);
					rPalm = mindist + 40;
					circle(FrameImg, mc[i], rPalm, Scalar(255, 0, 0), 2, 8, 0);
				}
				if (modeSelection.size() == 2){
					if (modeSelection[0].y > modeSelection[1].y){ Pstmc = modeSelection[1]; }
					else{ Pstmc = modeSelection[0]; }
				}
				else if (modeSelection.size() == 1){
					if (modeSelection[0].x > FrameImg.cols - 390 && modeSelection[0].x < FrameImg.cols - 320 && modeSelection[0].y>0 && modeSelection[0].y < 80){
						putText(FrameImg, "Blue color selected", Point(FrameImg.rows - 200, FrameImg.cols - 200), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(200, 0, 0), 2, CV_AA);
						lineColor = Scalar(255, 0, 0);
						lineSize = 6;

					}
					else if (modeSelection[0].x > FrameImg.cols - 310 && modeSelection[0].x < FrameImg.cols - 240 && modeSelection[0].y>0 && modeSelection[0].y < 80){
						putText(FrameImg, "green color selected", Point(FrameImg.rows - 200, FrameImg.cols - 200), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 200, 0), 2, CV_AA);
						lineColor = Scalar(0, 255, 0);
						lineSize = 6;
					}
					else if (modeSelection[0].x > FrameImg.cols - 230 && modeSelection[0].x < FrameImg.cols - 160 && modeSelection[0].y>0 && modeSelection[0].y < 80){
						putText(FrameImg, "red color selected", Point(FrameImg.rows - 200, FrameImg.cols - 200), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);
						lineColor = Scalar(0, 0, 255);
						lineSize = 6;
					}
					else if (modeSelection[0].x > FrameImg.cols - 150 && modeSelection[0].x < FrameImg.cols - 80 && modeSelection[0].y>0 && modeSelection[0].y < 80){
						putText(FrameImg, "Erase mode", Point(FrameImg.rows - 200, FrameImg.cols - 200), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);
						lineColor = Scalar(255, 255, 255);
						lineSize = 70;
					}
					else if (modeSelection[0].x > FrameImg.cols - 70 && modeSelection[0].x < FrameImg.cols && modeSelection[0].y>0 && modeSelection[0].y < 80){
						lineColor = Scalar(255, 255, 255);
						putText(FrameImg, "Clearing Screen...", Point(FrameImg.rows / 2, FrameImg.cols / 2), FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 200), 2, CV_AA);
						WhiteBoard = Mat(height, width, CV_8UC3, Scalar(255, 255, 255));
					}
					//line(WhiteBoard, Pstmc, modeSelection[0], lineColor, lineSize, 8, 0);
					if ((Pstmc.y != 0) && (Pstmc.x != 0))	{
						line(WhiteBoard, Pstmc, modeSelection[0], lineColor, lineSize);
						//line(WhiteBoard, Pstmc, modeSelection[0], lineColor, lineSize);
						//temp.copyTo(WhiteBoard);
					}
					Pstmc = modeSelection[0];
				}
			}
		}
		bitwise_and(FrameImg, WhiteBoard, FrameImg);
		
		/* Drawing buttons for mode */
		rectangle(FrameImg, Point(FrameImg.cols - 390, 0), Point(FrameImg.cols - 320, 80), Scalar(100, 0, 0), CV_FILLED, 8, 0); // blue
		rectangle(FrameImg, Point(FrameImg.cols - 310, 0), Point(FrameImg.cols - 240, 80), Scalar(0, 100, 0), CV_FILLED, 8, 0); // greeen
		rectangle(FrameImg, Point(FrameImg.cols - 230, 0), Point(FrameImg.cols - 160, 80), Scalar(0, 0, 100), CV_FILLED, 8, 0); // red
		rectangle(FrameImg, Point(FrameImg.cols - 150, 0), Point(FrameImg.cols - 80, 80), Scalar(255, 255, 255), CV_FILLED, 8, 0); // erase
		putText(FrameImg, "ERASE", Point(FrameImg.cols - 150, 50), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 2, CV_AA);
		rectangle(FrameImg, Point(FrameImg.cols - 70, 0), Point(FrameImg.cols, 80), Scalar(255, 255, 255), CV_FILLED, 8, 0); // clear
		putText(FrameImg, "CLEAR", Point(FrameImg.cols - 70, 50), FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2, CV_AA);

		imshow("whiteboard", WhiteBoard);
		imshow("Original with Contours", FrameImg);
	}
	cap.~VideoCapture();
	destroyAllWindows();
}

void HandGestureRecognition() {

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
		//imshow("Background", MaskImg);

		/* find face using haar-cascade */
		if (!face_cascade.empty()) {
			face_cascade.detectMultiScale(FrameImg, faces, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		}

		/* skin detection and find skin contours */
		vector<Vec4i> hierarchy;
		skinDetection(MaskImg, SkinImg, faces); //Skin Image contains the detected skin

		/// DISTANCE TRANSFORM*******************************************************************************
		Mat dist;
		distanceTransform(SkinImg, dist, CV_DIST_L2, 3);
		double minVal, maxVal;
		Point minLoc, maxLoc;
		minMaxLoc(dist, &minVal, &maxVal, &minLoc, &maxLoc);
		circle(FrameImg, maxLoc, 3, Scalar(0, 0, 255), 2, 8, 0);
		/// END OF DISTANCE TRANSFORM

		//Then define your mask image
		cv::Mat mask = cv::Mat::zeros(SkinImg.size(), SkinImg.type());

		//I assume you want to draw the circle at the center of your image, with a radius of 50
		rectangle(mask, Point(maxLoc.x - 5 * maxVal, maxLoc.y - 5 * maxVal), Point(maxLoc.x + 5 * maxVal, maxLoc.y + 1 * maxVal), Scalar(255, 255, 255), -1, 8, 0);
		//circle(mask, maxLoc, 3*maxVal, cv::Scalar(255, 255,255), -1, 8, 0);
		bitwise_and(SkinImg, mask, SkinImg);
		imshow("mask", SkinImg);


		findHandContour(contours, SkinImg.clone(), hierarchy); //contours is the vector of contour(vector of points)

		mu = vector<Moments>(contours.size());// moments; used for centroid of the centroid.
		mc = vector<Point>(contours.size());  // centeroid of contour 
		if (!Pstmc.size())	{
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
				mu[i] = moments(contours[i]);
				mc[i] = Point(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
				minEnclosingCircle(contours[i], enc_cnt[i], enc_r[i]);
				int radiusIn = findPalmCircleRad(MaskImg, contours[i], maxLoc);
				circle(FrameImg, enc_cnt[i], enc_r[i], Scalar(159, 0, 255), 2, 8, 0);
				circle(FrameImg, maxLoc, radiusIn, Scalar(0, 50, 0), 2, 8, 0);

				approxPolyDP(Mat(contours[i]), contours[i], arcLength(Mat(contours[i]), true)*0.032, true);
				drawContours(FrameImg, contours, i, Scalar(0, 255, 0), 2, 8, hierarchy);
				vector<float> pointAngle = vector<float>(contours[i].size());
				vector<int> distVec = vector<int>(contours[i].size());

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
						if (angle <= 90 && depth>radiusIn && depth<enc_r[i] && contours[i][far_indx].y <= maxLoc.y + 10)   {
							numConvex++;
							circle(FrameImg, ptFar, 5, Scalar(0, 0, 255), -1);
							circle(FrameImg, ptStart, 5, Scalar(255, 255, 255), -1);
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
						int dist = sqrt((maxLoc.x - contours[i][j].x) * (maxLoc.x - contours[i][j].x) + (maxLoc.y - contours[i][j].y) * (maxLoc.y - contours[i][j].y));
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
						if (pointAngle[j] > 0 && pointAngle[j] < 60 && dist >= 2*radiusIn && midPnt.y <= maxLoc.y + 20)  {
							line(FrameImg, maxLoc, midPnt, Scalar(0, 0, 0), 2, 8, 0);
							numless90++;
						}
					}
					putText(FrameImg, to_string(numless90), Point(50, 100), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(200, 0, 200), 2, CV_AA);
					rPalm = mindist + 40;
				}
				int xdiff = mc[i].x - Pstmc[i].x;
				int ydiff = mc[i].y - Pstmc[i].y;
				int distance = sqrt(xdiff*xdiff + ydiff*ydiff);

				/* classification of open and close hand using min-enclosing circle and palm circle */
				float diffR = enc_r[i] / radiusIn;
				if (diffR > 2.2 && (numless90 == 4 || numless90 == 5))	{
					putText(FrameImg, "OPEN HAND", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);

					/* mouse wheel up and down */
					int ydiff = mc[i].y - Pstmc[i].y;
					int threMovement = 20;
					cout << ydiff << '\t' << mc[i].y << '\t' << Pstmc[i].y << endl;
					if (ydiff > threMovement)	{
						TCHAR szExeFileName[MAX_PATH];
						GetModuleFileName(NULL, szExeFileName, MAX_PATH);
						mouse_event(MOUSEEVENTF_WHEEL, 0, 0, (-1)*WHEEL_DELTA, NULL);
					}
					else if (ydiff < (-1)* threMovement)	{
						TCHAR szExeFileName[MAX_PATH];
						GetModuleFileName(NULL, szExeFileName, MAX_PATH);
						mouse_event(MOUSEEVENTF_WHEEL, 0, 0, WHEEL_DELTA, NULL);
					}
				}
				else if (diffR < 1.6 || (numless90 == 0))	{
					putText(FrameImg, "CLOSED HAND", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);
				}
				else if ((numless90 == 1) && diffR >= 1.85)   {
					putText(FrameImg, "ONE", Point(500, 50), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);

					/*mouse left-up*/
					INPUT Input = { 0 };
					Input.type = INPUT_MOUSE;
					Input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
					::SendInput(1, &Input, sizeof(INPUT));

					/*moving cursor*/
					POINT cursorPos;
					GetCursorPos(&cursorPos);
					SetCursorPos(cursorPos.x + xdiff*distance / 3, cursorPos.y + ydiff*distance / 3);
				}
				else if (numless90 == 2 && diffR >= 1.85){
					putText(FrameImg, "TWO", Point(350, 50), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);
					INPUT Input = { 0 };
					Input.type = INPUT_MOUSE;
					Input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
					::SendInput(1, &Input, sizeof(INPUT));

					/*moving cursor*/
					POINT cursorPos;
					GetCursorPos(&cursorPos);
					SetCursorPos(cursorPos.x + xdiff*distance / 6, cursorPos.y + ydiff*distance / 6);
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
	vector<Point> mc; /* Current and Previous Hand center */
	vector<Point2f> enc_cnt; /* Enclosing circle center */
	vector<float> enc_r; /* Enclosing circle radius */
	vector<vector<Point>> fingers;  /* Detected fingers positions */
	CascadeClassifier face_cascade;
	VideoCapture cap;
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

		/// DISTANCE TRANSFORM*******************************************************************************
		Mat dist;
		distanceTransform(SkinImg, dist, CV_DIST_L2, 3);
		double minVal, maxVal;
		Point minLoc, maxLoc;
		minMaxLoc(dist, &minVal, &maxVal, &minLoc, &maxLoc);
		circle(FrameImg, maxLoc, 3, Scalar(0, 0, 255), 2, 8, 0);
		/// END OF DISTANCE TRANSFORM

		//Then define your mask image
		cv::Mat mask = cv::Mat::zeros(SkinImg.size(), SkinImg.type());

		//I assume you want to draw the circle at the center of your image, with a radius of 50
		rectangle(mask, Point(maxLoc.x - 5 * maxVal, maxLoc.y - 5 * maxVal), Point(maxLoc.x + 5 * maxVal, maxLoc.y + 1 * maxVal), Scalar(255, 255, 255), -1, 8, 0);
		//circle(mask, maxLoc, 3*maxVal, cv::Scalar(255, 255,255), -1, 8, 0);
		bitwise_and(SkinImg, mask, SkinImg);
		imshow("mask", SkinImg);

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
				mu[i] = moments(contours[i]);
				mc[i] = Point(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
				minEnclosingCircle(contours[i], enc_cnt[i], enc_r[i]);
				int radiusIn = findPalmCircleRad(MaskImg, contours[i], maxLoc);
				circle(FrameImg, enc_cnt[i], enc_r[i], Scalar(159, 0, 255), 2, 8, 0);
				circle(FrameImg, maxLoc, radiusIn, Scalar(0, 50, 0), 2, 8, 0);

				approxPolyDP(Mat(contours[i]), contours[i], arcLength(Mat(contours[i]), true)*0.032, true);
				drawContours(FrameImg, contours, i, Scalar(0, 255, 0), 2, 8, hierarchy);
				vector<float> pointAngle = vector<float>(contours[i].size());
				vector<int> distVec = vector<int>(contours[i].size());

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
						if (angle <= 90 && depth>radiusIn && depth<enc_r[i] && contours[i][far_indx].y <= maxLoc.y + 10)   {
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
						int dist = sqrt((maxLoc.x - contours[i][j].x) * (maxLoc.x - contours[i][j].x) + (maxLoc.y - contours[i][j].y) * (maxLoc.y - contours[i][j].y));
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
						if (pointAngle[j] > 0 && pointAngle[j] < 60 && dist >= 1.5*radiusIn && midPnt.y <= maxLoc.y + 20)  {
							line(FrameImg, maxLoc, midPnt, Scalar(0, 0, 0), 2, 8, 0);
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
				float diffR = enc_r[i] / radiusIn;
				//putText(FrameImg, to_string(diffR), Point(50, 300), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);

				//putText(FrameImg, to_string(numConvex), Point(50, 150), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 0, 0), 2, CV_AA);
				//putText(FrameImg, to_string(diffR), Point(50, 200), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 2, CV_AA);
				/*if (circles.size() == 0 && diffR<1.6 && (numless90 == 0) && (numConvex == 0)) {
					putText(FrameImg, "CLOSE PALM", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);
				}
				else if (circles.size() == 0 && (diffR>2.1 && diffR<2.9) && (numless90 <= 3) && (numConvex == 0)) {
					putText(FrameImg, "OPEN PALM", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);
				}
				else*/ if (circles.size()>0 && (numless90 == 1 || numless90 == 2)){
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
			}
		}

		imshow("Skin", SkinImg);
		imshow("Original with Contours", FrameImg);
	}
	cap.~VideoCapture();
	destroyAllWindows();
}


void skinDetection(Mat &FrameImg, Mat &SkinImg, vector<Rect> &faces) {
	Mat tempImg;
	cvtColor(FrameImg, tempImg, CV_BGR2YCrCb);
	inRange(tempImg, Scalar(0, 133, 77), Scalar(255, 173, 127), SkinImg); // Jaseong
	//inRange(tempImg, Scalar(0, 158, 85), Scalar(255, 180, 115), SkinImg); //Praful
	dilate(SkinImg, SkinImg, Mat());
	erode(SkinImg, SkinImg, Mat());
	GaussianBlur(SkinImg, SkinImg, Size(9, 9), 0);
	medianBlur(SkinImg, SkinImg, 13);
	threshold(SkinImg, SkinImg, 0, 255, THRESH_BINARY + THRESH_OTSU);

	if (!faces.size()) {}
	else {
		for (int i = 0; i < faces.size(); i++) {
			Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.8);
			ellipse(SkinImg, center, Size(faces[i].width*0.65, faces[i].height), 0, 0, 360, Scalar(0, 0, 0), -1, 8, 0);
		}
	}
}

void findHandContour(vector<vector<Point>>& contours, Mat& SkinImg, vector<Vec4i> hierarchy)  {

	findContours(SkinImg, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	for (vector<vector<Point> >::iterator it = contours.begin(); it != contours.end();) {
		if (it->size()<105)
			it = contours.erase(it);
		else
			++it;
	}
}

int findPalmCircleRad(Mat FrameImg, vector<Point> contours, Point Hand_cnt)
{
	int n;
	Point min_point;
	int mindist = FrameImg.size().height;

	if (!contours.size())
		return 0;
	else {
		for (int i = 0; i < contours.size(); i = i + 10) {
			int dist = 0;
			int cx = Hand_cnt.x;
			int cy = Hand_cnt.y;

			dist = sqrt((cx - contours[i].x)*(cx - contours[i].x) + (cy - contours[i].y)*(cy - contours[i].y));
			if (dist <= mindist && dist > 0)  {
				mindist = dist;
			}
			min_point = contours[i];
		}
		return mindist;
	}
}

Point2f point;
bool addRemovePt = false;

void onMouse(int event, int x, int y, int /*flags*/, void* /*param*/)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		point = Point2f((float)x, (float)y);
		addRemovePt = true;
	}
}

void PacmanGame()
{
	VideoCapture cap;
	int flag = 1;
	TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
	clock_t begin, end;
	Size subPixWinSize(10, 10), winSize(31, 31);
	Mat pacman = imread("pacman.jpg");
	Mat apple = imread("apple.png");

	if (pacman.empty()){
		cout << "Can't load image..." << endl;
	}
	resize(pacman, pacman, pacman.size() / 3, 0.5, 0.5, INTER_LINEAR);
	resize(apple, apple, apple.size() / 120, 0.5, 0.5, INTER_LINEAR);

	const int MAX_COUNT = 1;
	const int RND_COUNT = 20;
	bool needToInit = false;
	bool nightMode = false;

	cap.open(0);
	if (!cap.isOpened()){
		cout << "Could not initialize capturing...\n";
	}
	//setMouseCallback("Pacman", onMouse, 0);

	Mat gray, prevGray, image, frame;
	vector<Point2f> points[2];
	vector<Point> rndPnt;
	vector<float> rndAngles;
	float prevAngle = 0, currAngle = 0;
	int radius = 30, radius_ball = 9;
	int distThrsh = radius + radius_ball;
	int numBalls = RND_COUNT*MAX_COUNT;
	int score = 0;

	for (;;)
	{
		cap >> frame;
		if (frame.empty())
			break;
		flip(frame, frame, 1);
		frame.copyTo(image);
		cvtColor(image, gray, COLOR_BGR2GRAY);

		if (nightMode)
			image = Scalar::all(0);

		if (needToInit)
		{
			// automatic initialization
			goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, false, 0.04);
			//cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
			if (!rndPnt.empty() || !rndAngles.empty()){
				rndPnt.clear();
				rndAngles.clear();
			}
			while (rndPnt.size() < MAX_COUNT*RND_COUNT){
				for (int i = 0; i < points[1].size(); i++){
					int rndx = rand() % frame.cols;
					int rndy = rand() % frame.rows;
					float rndAngle = rand() % 360;
					int dist = sqrt(pow((points[1][i].x - rndx), 2) + pow((points[1][i].y - rndy), 2));

					if (dist >= radius + 10)
						rndPnt.push_back(Point(rndx, rndy));
					rndAngles.push_back(rndAngle);
				}
			}
			score = 0;
			nightMode = false;
			addRemovePt = false;
			begin = clock();
		}
		else if (!points[0].empty())
		{
			vector<uchar> status;
			vector<float> err;
			if (prevGray.empty())
				gray.copyTo(prevGray);
			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize, 3, termcrit, 0, 1e-4);

			size_t i, k;
			for (i = k = 0; i < points[1].size(); i++){
				if (points[1][i].y < 0){
					points[1][i].y = 0;
				}
				else if (points[1][i].y > frame.rows - pacman.rows){
					points[1][i].y = frame.rows - pacman.rows;
				}

				if (points[1][i].x < 0){
					points[1][i].x = 0;
				}
				else if (points[1][i].x>frame.cols - pacman.cols){
					points[1][i].x = frame.cols - pacman.cols;
				}
				else{
					points[1][k++] = points[1][i];
				}
				pacman.copyTo(image(Rect(points[1][i].x, points[1][i].y, pacman.cols, pacman.rows)));
				for (int j = 0; j < rndPnt.size(); j++){
					rndPnt[j].x = rndPnt[j].x + 5 * cos(rndAngles[j] * (3.14 / 180));
					rndPnt[j].y = rndPnt[j].y + 5 * sin(rndAngles[j] * (3.14 / 180));
					if (rndPnt[j].y <  0){
						rndAngles[j] = 360 - rndAngles[j];
						rndPnt[j].y = 0;
					}
					else if (rndPnt[j].y > frame.rows - apple.rows){
						rndAngles[j] = 360 - rndAngles[j];
						rndPnt[j].y = frame.rows - apple.rows;
					}

					if (rndPnt[j].x <  0){
						if (rndAngles[j]>180)
							rndAngles[j] = 540 - rndAngles[j];
						else{ rndAngles[j] = 180 - rndAngles[j]; }
						rndPnt[j].x = 0;
					}
					else if (rndPnt[j].x > frame.cols - apple.cols){
						if (rndAngles[j]>180)
							rndAngles[j] = 540 - rndAngles[j];
						else{ rndAngles[j] = 180 - rndAngles[j]; }
						rndPnt[j].x = frame.cols - apple.cols;
					}

					int tmpDist = sqrt(pow(points[1][i].x - rndPnt[j].x, 2) + pow(points[1][i].y - rndPnt[j].y, 2));
					if (tmpDist > distThrsh)
						apple.copyTo(image(Rect(rndPnt[j].x, rndPnt[j].y, apple.cols, apple.rows)));
					else{
						rndPnt.erase(rndPnt.begin() + j);
						score++;
					}
				}

				float dist = sqrt(pow((points[1][i].x - points[0][i].x), 2) + pow((points[1][i].y - points[0][i].y), 2));
				if (dist > 2){
					currAngle = atan2(points[0][i].y - points[1][i].y, points[1][i].x - points[0][i].x);
				}
				else{
					currAngle = 0;
				}
			}
			points[1].resize(k);
		}

		if (score < numBalls){
			end = clock();
			int elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
			putText(image, "Time Elapsed: ", Point(300, 25), FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, CV_AA);
			putText(image, to_string(elapsed_secs), Point(540, 27), FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, CV_AA);
			putText(image, "sec", Point(580, 25), FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, CV_AA);
			putText(image, "score: ", Point(0, 25), FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, CV_AA);
			putText(image, to_string(score), Point(100, 27), FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, CV_AA);

		}
		else{

			if (flag == 1)
			{
				end = clock();
				flag++;
			}
			else{
				int elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
				nightMode = true;
				if (elapsed_secs <= 25)
				putText(image, "GOOD JOB", Point(25, image.rows / 2), FONT_HERSHEY_DUPLEX, 3.5, cv::Scalar(255, 255, 255), 2, CV_AA);
				else { putText(image, "You can do a better job", Point(25, image.rows / 2), FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 255, 255), 2, CV_AA); }
				putText(image, "Elapsed time:", Point(image.cols / 2 - 250, image.rows / 2 + 100), FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 255, 255), 2, CV_AA);
				putText(image, to_string(elapsed_secs), Point(image.cols / 2 - 10, image.rows / 2 + 100), FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 255, 255), 2, CV_AA);
				putText(image, "sec", Point(image.cols / 2 + 40, image.rows / 2 + 100), FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 255, 255), 2, CV_AA);
				putText(image, "Press 'r' to play again", Point(image.cols / 2 - 110, image.rows / 2 + 150), FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 255, 255), 2, CV_AA);
				putText(image, "Press 'esc' to quit", Point(image.cols / 2 - 110, image.rows / 2 + 200), FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 255, 255), 2, CV_AA);
			}
		}


		if (addRemovePt && points[1].size() < (size_t)MAX_COUNT){
			vector<Point2f> tmp;
			tmp.push_back(point);
			points[1].push_back(tmp[0]);
			addRemovePt = false;
		}

		needToInit = false;
		prevAngle = currAngle;
		imshow("LK Demo", image);

		char c = (char)waitKey(10);
		if (c == 27){
			destroyAllWindows();
			break;
		}
			
		switch (c){
		case 'r':
			needToInit = true;
			break;
		case 'c':
			points[0].clear();
			points[1].clear();
			rndPnt.clear();
			break;
		case 'n':
			nightMode = !nightMode;
			break;
		}

		std::swap(points[1], points[0]);
		cv::swap(prevGray, gray);
	}
}