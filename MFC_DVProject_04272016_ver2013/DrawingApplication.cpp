#include "Header.h"
#include "stdafx.h"

using namespace cv;
using namespace std;

void BackgroundImgSubtract(Mat &, Mat &, Mat &);
void skinDetection(Mat &, Mat &, vector<Rect>&);
void findHandContour(vector<vector<Point>>&, Mat&, vector<Vec4i>);
int findPalmCircleRad(Mat, vector<Point>, Point);

void DrawingApplication(int argc, char *argv[]) {

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

	/* getting the size of laptop */
	RECT laptop;
	GetWindowRect(GetDesktopWindow(), &laptop);

	face_cascade.load("C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml");
	if (face_cascade.empty()) {
		cout << "face_cascade xml file is failed  " << face_cascade.empty() << endl;
		return 0;
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
		findHandContour(contours, SkinImg.clone(), hierarchy); //contours is the vector of contour(vector of points)

		mu = vector<Moments>(contours.size());// moments; used for centroid of the centroid.
		mc = vector<Point>(contours.size());  // center/oid of contour 
		if (!Pstmc.size()){
			Pstmc = vector<Point>(contours.size());
		}
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
				vector<float> pointAngle = vector<float>(contours[i].size());
				vector<int> distVec = vector<int>(contours[i].size());
				mu[i] = moments(contours[i]);
				mc[i] = Point(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
				minEnclosingCircle(contours[i], enc_cnt[i], enc_r[i]);
				int radiusIn = findPalmCircleRad(MaskImg, contours[i], mc[i]);

				approxPolyDP(Mat(contours[i]), contours[i], arcLength(Mat(contours[i]), true)*0.04, true);
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
						if (depth>radiusIn && depth<enc_r[i] && contours[i][far_indx].y <= mc[i].y + 10)   {
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

				if (modeSelection.size() == 1){
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
						white = Mat(height, width, CV_8UC3, Scalar(255, 255, 255));
					}
				}
				else if (modeSelection.size() == 2){
					if (modeSelection[0].y>modeSelection[1].y)
						Pstmc[i] = modeSelection[1];
					else{ Pstmc[i] = modeSelection[0]; }
				}
				if (modeSelection.size() == 1){

					if (Pstmc[i].x != 0 && Pstmc[i].y != 0)
					{
						line(white, Pstmc[i], modeSelection[0], lineColor, lineSize);
					}

					Pstmc[i] = modeSelection[0];
				}
			}
		}
		bitwise_and(FrameImg, white, FrameImg);
		/* Drawing buttons for mode */
		rectangle(FrameImg, Point(FrameImg.cols - 390, 0), Point(FrameImg.cols - 320, 80), Scalar(100, 0, 0), CV_FILLED, 8, 0); // blue
		rectangle(FrameImg, Point(FrameImg.cols - 310, 0), Point(FrameImg.cols - 240, 80), Scalar(0, 100, 0), CV_FILLED, 8, 0); // greeen
		rectangle(FrameImg, Point(FrameImg.cols - 230, 0), Point(FrameImg.cols - 160, 80), Scalar(0, 0, 100), CV_FILLED, 8, 0); // red
		rectangle(FrameImg, Point(FrameImg.cols - 150, 0), Point(FrameImg.cols - 80, 80), Scalar(255, 255, 255), CV_FILLED, 8, 0); // erase
		putText(FrameImg, "ERASE", Point(FrameImg.cols - 150, 50), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 2, CV_AA);
		rectangle(FrameImg, Point(FrameImg.cols - 70, 0), Point(FrameImg.cols, 80), Scalar(255, 255, 255), CV_FILLED, 8, 0); // clear
		putText(FrameImg, "CLEAR", Point(FrameImg.cols - 70, 50), FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2, CV_AA);

		imshow("whiteboard", white);
		imshow("Original with Contours", FrameImg);
	}
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
