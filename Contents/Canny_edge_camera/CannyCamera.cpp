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

	//������ͷ
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

		//����տ�ʼ��ͼƬ

		srcImage = frame.clone();

		//������frameͬ���ͺʹ�С�ľ���
		nImage.create(frame.size(),frame.type()); 

		//��Ϊ�Ҷ�ͼ
		cvtColor(frame, frame, COLOR_BGR2GRAY);
		
		//Canny��Ե���
		on_canny(0,0); //�ص�����

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
	//��˹ƽ��
	GaussianBlur(frame,frame,Size(7, 7), 0,0);
	//��Ե���
	Canny(frame, frame,cannyLowThreshold, cannyLowThreshold*3, 3);
	//�Ƚ�nImage�ڵ�����Ԫ������Ϊ0
	nImage = Scalar::all(0);
	//ʹ��Canny��������ı�Եͼframe��Ϊ���룬��ԭͼsrcImage������Ŀ��ͼnImage��
	srcImage.copyTo(nImage,frame);

	imshow(window_name, nImage);
}
