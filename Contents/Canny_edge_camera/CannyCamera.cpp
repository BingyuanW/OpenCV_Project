//https://docs.opencv.org/master/da/d6a/tutorial_trackbar.html

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include<opencv2/imgproc.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

string window_name = "video | q or esc to quit";

int cannyLowThreshold = 1;

Mat frame;
Mat srcImage;
Mat nImage;

static void on_canny(int, void*);

int main() {

	int n = 0;
    char filename[200];

	namedWindow(window_name, WINDOW_AUTOSIZE); 

	VideoCapture capture;

	//打开摄像头
	capture.open(0); 

	if (!capture.isOpened()) {
		cerr << "Failed to open the video device!\n" << endl;
		return 1;
	}

	cout << "press space to save a picture. q or esc to quit" << endl;
	createTrackbar("CannyThreshold", window_name, &cannyLowThreshold, 100);
	for (;;) {
		capture >> frame; 
		if (frame.empty())
			break;

		//保存刚开始的图片

		srcImage = frame.clone();

		//创建与frame同类型和大小的矩阵
		nImage.create(frame.size(),frame.type()); 

		//变为灰度图
		cvtColor(frame, frame, COLOR_BGR2GRAY);
		
		//Canny边缘检测
		on_canny(0,0); //回调函数

		char key = (char)waitKey(30); //delay N millis, usually long enough to display and capture input

		switch (key) {
		case 'q':
		case 'Q':
		case 27: //escape key
			return 0;
		case ' ': //Save an image
			sprintf(filename,"filename%.3d.jpg",n++);	
			imwrite(filename, nImage);
			cout << "Saved " << filename << endl;
			break;
		default:
			break;
		}
	}
	capture.release();

	return 0;

}

void on_canny(int, void*){
	//高斯平滑
	GaussianBlur(frame,frame,Size(7, 7), 0,0);
	//边缘检测
	Canny(frame, frame,cannyLowThreshold, cannyLowThreshold*3, 3);
	//先将nImage内的所有元素设置为0
	nImage = Scalar::all(0);
	//使用Canny算子输出的边缘图frame作为掩码，将原图srcImage拷贝到目标图nImage中
	srcImage.copyTo(nImage,frame);

	imshow(window_name, nImage);
}
