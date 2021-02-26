/*
https://blog.csdn.net/qq_38231713/article/details/
90648139?utm_medium=distribute.pc_relevant.none-ta
sk-blog-BlogCommendFromMachineLearnPai2-3.control&di
st_request_id=28c2fbca-cb78-4e3b-be9b-dc46494e42da&de
pth_1-utm_source=distribute.pc_relevant.none-task-blo
g-BlogCommendFromMachineLearnPai2-3.control

*/
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include<opencv2/imgproc.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

//捕获图片
int picture_capture() {
	char filename[] = "pic_1.jpg";

	string window_name = "video | q or esc to quit";
	namedWindow(window_name, WINDOW_KEEPRATIO); //resizable window;

	Mat frame;
	VideoCapture capture;

	capture.open(0); //打开摄像头

	if (!capture.isOpened()) {
		cerr << "Failed to open the video device!\n" << endl;
		return 1;
	}

	cout << "press space to save a picture. q or esc to quit" << endl;
	//return process(capture);
	for (;;) {
		capture >> frame; 
		if (frame.empty())
			break;

		imshow(window_name, frame);
		char key = (char)waitKey(30); //delay N millis, usually long enough to display and capture input

		switch (key) {
		case 'q':
		case 'Q':
		case 27: //escape key
			capture.release();
			return 0;
		case ' ': //Save an image	
			imwrite(filename, frame);
			cout << "Saved " << filename << endl;
			break;
		default:
			break;
		}
	}
	return 0;
}

//生成高斯卷积核 kernel
void Gaussian_kernel(int kernel_size, int sigma, Mat &kernel)
{
	const double PI = 3.1415926;
	int m = kernel_size / 2;

	kernel = Mat(kernel_size, kernel_size, CV_32FC1);
	float s = 2 * sigma * sigma;
	//笔记： 注意二维高斯函数下面就是2*pi*sigma^2，没有根号
	for (int i = 0; i < kernel_size; i++)
	{
		for (int j = 0; j < kernel_size; j++)
		{
			int x = i - m;
			int y = j - m;

			kernel.at<float>(i, j) = exp(-(x * x + y * y) / s) / (PI * s);
		}
	}
}

/*
计算梯度值和方向
imageSource 原始灰度图
imageX X方向梯度图像
imageY Y方向梯度图像
gradXY 该点的梯度幅值
theta 梯度方向角度 theta=arctan(imageY/imageX)
*/
void GradDirection(const Mat imageSource, Mat &imageX, Mat &imageY, Mat &gradXY, Mat &theta)
{

	imageX = Mat::zeros(imageSource.size(), CV_32SC1);
	imageY = Mat::zeros(imageSource.size(), CV_32SC1);
	gradXY = Mat::zeros(imageSource.size(), CV_32SC1);
	theta = Mat::zeros(imageSource.size(), CV_32SC1);

	int rows = imageSource.rows;
	int cols = imageSource.cols;

	/*
	Mat.step参数指图像的一行实际占用的内存长度，以字节为基本单位，
	因为opencv中的图像会对每行的长度自动补齐（8的倍数），
	编程时尽量使用指针，指针读写像素是速度最快的，使用at函数最慢。
	*/
	int stepXY = imageX.step;
	int step = imageSource.step;

	/*
	Mat::data的默认类型为uchar*，但很多时候需要处理其它类型，如float、int，
	此时需要将data强制类型转换，如：
	Mat src(1000,1000,CV_32F);
	float* myptr = (float*)src.data;
	无论Mat的type是何种类型，Mat::data均为uchar*
	*/
	uchar *PX = imageX.data;
	uchar *PY = imageY.data;
	uchar *P = imageSource.data;
	uchar *XY = gradXY.data;
	for (int i = 1; i < rows - 1; i++)
	{
		for (int j = 1; j < cols - 1; j++)
		{
			int a00 = P[(i - 1)*step + j - 1];
			int a01 = P[(i - 1)*step + j];
			int a02 = P[(i - 1)*step + j + 1];

			int a10 = P[i*step + j - 1];
			int a11 = P[i*step + j];
			int a12 = P[i*step + j + 1];

			int a20 = P[(i + 1)*step + j - 1];
			int a21 = P[(i + 1)*step + j];
			int a22 = P[(i + 1)*step + j + 1];

			double gradX = double(a02 + 2 * a12 + a22 - a00 - 2 * a10 - a20);
			double gradY = double(a00 + 2 * a01 + a02 - a20 - 2 * a21 - a22);

			//遍历是从i,j=1开始到rows,cols-2的，图像像素都初始化为0，
			//所以最外边的一圈像素为0是黑的
			//PX[i*stepXY + j*(stepXY / step)] = abs(gradX);
			//PY[i*stepXY + j*(stepXY / step)] = abs(gradY);
			imageX.at<int>(i, j) = abs(gradX);
			imageY.at<int>(i, j) = abs(gradY);
			if (gradX == 0)
			{
				gradX = 0.000000000001;
			}
			//弧度数改为角度数,除以pi乘以180
			theta.at<int>(i, j) = atan(gradY / gradX) * 57.3;
			theta.at<int>(i, j) = (theta.at<int>(i, j) + 360) % 360;
			gradXY.at<int>(i, j) = sqrt(gradX*gradX + gradY * gradY);
			//XY[i * stepXY + j * (stepXY / step)] = sqrt(gradX * gradX + gradY * gradY);
		}
	}
	/*
	在经过处理后，需要用convertScaleAbs()函数将其转回原来的uint8形式，
	否则将无法显示图像，而只是一副灰色的窗口。
	函数原型为
	void convertScaleAbs(InputArray src, OutputArray dst,
								  double alpha = 1, double beta = 0);
	其中可选参数alpha是伸缩系数，beta是加到结果上的一个值，结果返回uint8类型的图片
	功能：实现将原图片转换为uint8类型
	*/
	convertScaleAbs(imageX, imageX);
	convertScaleAbs(imageY, imageY);
	convertScaleAbs(gradXY, gradXY);
}

/*
局部非极大值抑制
沿着该点梯度方向，比较前后两个点的幅值大小，若该点大于前后两点，则保留，
若该点小于前后两点任意一点，则置为0；
imageInput 输入得到梯度图像
imageOutput 输出的非极大值抑制图像
theta 每个像素点的梯度方向角度
imageX X方向梯度
imageY Y方向梯度
*/
void NonLocalMaxValue(const Mat imageInput, Mat &imageOutput, const Mat &theta, const Mat &imageX, const Mat &imageY)
{
	imageOutput = imageInput.clone();


	int cols = imageInput.cols;
	int rows = imageInput.rows;

	for (int i = 1; i < rows - 1; i++)
	{
		for (int j = 1; j < cols - 1; j++)
		{
			//执行continue时，会执行本次迭代的剩余部分，并开始下一次迭代
			//执行break时，会执行包含它的循环，并执行下一阶段
			if (0 == imageInput.at<uchar>(i, j))continue;

			int g00 = imageInput.at<uchar>(i - 1, j - 1);
			int g01 = imageInput.at<uchar>(i - 1, j);
			int g02 = imageInput.at<uchar>(i - 1, j + 1);

			int g10 = imageInput.at<uchar>(i, j - 1);
			int g11 = imageInput.at<uchar>(i, j);
			int g12 = imageInput.at<uchar>(i, j + 1);

			int g20 = imageInput.at<uchar>(i + 1, j - 1);
			int g21 = imageInput.at<uchar>(i + 1, j);
			int g22 = imageInput.at<uchar>(i + 1, j + 1);

			int direction = theta.at<int>(i, j); //该点梯度的角度值
			int g1 = 0;
			int g2 = 0;
			int g3 = 0;
			int g4 = 0;
			double tmp1 = 0.0; //保存亚像素点插值得到的灰度数
			double tmp2 = 0.0;
			//在C中，有abs(),labs(),fabs()分别计算int ,long ,double 类型的绝对值；
			//而在C++中，abs()已被重载，可适用于各种类型
			double weight = fabs((double)imageY.at<uchar>(i, j) / (double)imageX.at<uchar>(i, j));

			//去掉也可以，如果weight==0，那direction==0或180就不用差值，直接用原来的值
			if (weight == 0)
				weight = 0.0000001;
			/*
			关于这些公式的含义
			https://www.cnblogs.com/love6tao/p/5152020.html 有解释
			形如g10 * (1 - weight) + g20 * (weight)都是用两边的像素计算得到的亚像素
			不过他w和1-w应该写反了
			*/
			if (weight > 1)
			{
				weight = 1 / weight;
			}
			if ((0 <= direction && direction < 45) || 180 <= direction && direction < 225)
			{
				tmp1 = g10 * (1 - weight) + g20 * (weight);
				tmp2 = g02 * (weight)+g12 * (1 - weight);
			}
			if ((45 <= direction && direction < 90) || 225 <= direction && direction < 270)
			{
				tmp1 = g01 * (1 - weight) + g02 * (weight);
				tmp2 = g20 * (weight)+g21 * (1 - weight);
			}
			if ((90 <= direction && direction < 135) || 270 <= direction && direction < 315)
			{
				tmp1 = g00 * (weight)+g01 * (1 - weight);
				tmp2 = g21 * (1 - weight) + g22 * (weight);
			}
			if ((135 <= direction && direction < 180) || 315 <= direction && direction < 360)
			{
				tmp1 = g00 * (weight)+g10 * (1 - weight);
				tmp2 = g12 * (1 - weight) + g22 * (weight);
			}

			if (imageInput.at<uchar>(i, j) < tmp1 || imageInput.at<uchar>(i, j) < tmp2)
			{
				imageOutput.at<uchar>(i, j) = 0;
			}
		}
	}

}

/*
双阈值的机理是：
指定一个低阈值A，一个高阈值B，选取占直方图总数70%为B，且B为1.5到2倍大小的A；
灰度值小于A的，置为0,灰度值大于B的，置为255；
*/
void DoubleThreshold(Mat &imageInput, const double lowThreshold, const double highThreshold)
{
	int cols = imageInput.cols;
	int rows = imageInput.rows;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			double temp = imageInput.at<uchar>(i, j);
			temp = temp > highThreshold ? (255) : (temp);
			temp = temp < lowThreshold ? (0) : (temp);
			imageInput.at<uchar>(i, j) = temp;
		}
	}

}


/*
连接处理:
灰度值介于A和B之间的，考察该像素点临近的8像素是否有灰度值为255的，
若没有255的，表示这是一个孤立的局部极大值点，予以排除，置为0；
若有255的，表示这是一个跟其他边缘有“接壤”的可造之材，置为255，
之后重复执行该步骤，直到考察完最后一个像素点。

其中的邻域跟踪算法，从值为255的像素点出发找到周围满足要求的点，把满足要求的点设置为255，
然后修改i,j的坐标值，i,j值进行回退，在改变后的i,j基础上继续寻找255周围满足要求的点。
当所有连接255的点修改完后，再把所有上面所说的局部极大值点置为0；（算法可以继续优化）。

参数1，imageInput：输入和输出的梯度图像
参数2，lowTh:低阈值
参数3，highTh:高阈值
*/
void DoubleThresholdLink(Mat &imageInput, double lowTh, double highTh)
{
	int cols = imageInput.cols;
	int rows = imageInput.rows;

	for (int i = 1; i < rows - 1; i++)
	{
		for (int j = 1; j < cols - 1; j++)
		{
			double pix = imageInput.at<uchar>(i, j);
			if (pix != 255)continue;
			bool change = false;
			for (int k = -1; k <= 1; k++)
			{
				for (int u = -1; u <= 1; u++)
				{
					if (k == 0 && u == 0)continue;
					double temp = imageInput.at<uchar>(i + k, j + u);
					if (temp >= lowTh && temp <= highTh)
					{
						imageInput.at<uchar>(i + k, j + u) = 255;
						change = true;
					}
				}
			}
			//如果要连接像素值，则如果连接的是上边三个或者左边紧挨着的一个
			//可能会对之前的像素产生影响，即可能前面的邻接像素也要改变
			//这里采取的方法是不管要改变哪个像素，都将当前位置返回到左上角
			//最上面一条边和最左边一条边不能返回到左上角，就分别左移和上移一格
			//注意j马上还要+1，i不用+1
			//至于为什么是i>1而j>2，其实j>1也可以，只是因为j是减2，而2-2=0，不过还要+1所以没问题
			if (change)
			{
				if (i > 1)i--;
				if (j > 2)j -= 2;

			}
		}
	}

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (imageInput.at<uchar>(i, j) != 255)
			{
				imageInput.at<uchar>(i, j) = 0;
			}
		}
	}
}



int main() {

	//捕获图片
	picture_capture();  

	//载入原图
	Mat srcImage, grayImage;
	srcImage = imread("pic_1.jpg");
	if (!srcImage.data) { cerr << "Failed to load image!\n" << endl; return 1; }

	//变为灰度图
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);

	//高斯滤波
	Mat gausKernel;
	int kernel_size = 5;
	double sigma = 1;
	Gaussian_kernel(kernel_size, sigma, gausKernel);

	Mat gausImage;
	//Convolves an image with kernel 即利用内核实现对图像的卷积运算
	filter2D(grayImage, gausImage, grayImage.depth(), gausKernel);
	//imshow("gaus image", gausImage);
	//waitKey(0);
	//imwrite("gausImage.jpg", gausImage);

	//计算XY方向梯度
	Mat imageX, imageY, imageXY;
	//theta为梯度方向，theta=arctan(imageY/imageX)
	Mat theta;
	GradDirection(gausImage, imageX, imageY, imageXY, theta);
	//imshow("XY grad", imageXY);
	//waitKey(0);
	//imwrite("GradImage.jpg", imageXY);

	//对梯度幅值进行非极大值抑制
	Mat localImage;
	NonLocalMaxValue(imageXY, localImage, theta, imageX, imageY);
	//imshow("Non local maxinum image", localImage);
	//waitKey(0);
	//imwrite("localImage.jpg", localImage);

	//双阈值算法检测和边缘连接
	DoubleThreshold(localImage, 60, 100);
	DoubleThresholdLink(localImage, 60, 100);
	imshow("canny image", localImage);
	waitKey(0);
	imwrite("cannyImage.jpg", localImage);

	return 0;

}


