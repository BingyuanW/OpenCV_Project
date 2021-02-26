/*
https://blog.csdn.net/qq_38231713/article/details/90648139?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.control&dist_request_id=28c2fbca-cb78-4e3b-be9b-dc46494e42da&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.control

*/
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include<opencv2/imgproc.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

//����ͼƬ
int picture_capture() {
	char filename[] = "pic_1.jpg";

	string window_name = "video | q or esc to quit";
	namedWindow(window_name, WINDOW_KEEPRATIO); //resizable window;

	Mat frame;
	VideoCapture capture;

	capture.open(0); //������ͷ

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

//���ɸ�˹����� kernel
void Gaussian_kernel(int kernel_size, int sigma, Mat &kernel)
{
	const double PI = 3.1415926;
	int m = kernel_size / 2;

	kernel = Mat(kernel_size, kernel_size, CV_32FC1);
	float s = 2 * sigma * sigma;
	//�ʼǣ� ע���ά��˹�����������2*pi*sigma^2��û�и���
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
�����ݶ�ֵ�ͷ���
imageSource ԭʼ�Ҷ�ͼ
imageX X�����ݶ�ͼ��
imageY Y�����ݶ�ͼ��
gradXY �õ���ݶȷ�ֵ
theta �ݶȷ���Ƕ� theta=arctan(imageY/imageX)
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
	Mat.step����ָͼ���һ��ʵ��ռ�õ��ڴ泤�ȣ����ֽ�Ϊ������λ��
	��Ϊopencv�е�ͼ����ÿ�еĳ����Զ����루8�ı�������
	���ʱ����ʹ��ָ�룬ָ���д�������ٶ����ģ�ʹ��at����������
	*/
	int stepXY = imageX.step;
	int step = imageSource.step;

	/*
	Mat::data��Ĭ������Ϊuchar*�����ܶ�ʱ����Ҫ�����������ͣ���float��int��
	��ʱ��Ҫ��dataǿ������ת�����磺
	Mat src(1000,1000,CV_32F);
	float* myptr = (float*)src.data;
	����Mat��type�Ǻ������ͣ�Mat::data��Ϊuchar*
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

			//�����Ǵ�i,j=1��ʼ��rows,cols-2�ģ�ͼ�����ض���ʼ��Ϊ0��
			//��������ߵ�һȦ����Ϊ0�Ǻڵ�
			//PX[i*stepXY + j*(stepXY / step)] = abs(gradX);
			//PY[i*stepXY + j*(stepXY / step)] = abs(gradY);
			imageX.at<int>(i, j) = abs(gradX);
			imageY.at<int>(i, j) = abs(gradY);
			if (gradX == 0)
			{
				gradX = 0.000000000001;
			}
			//��������Ϊ�Ƕ���,����pi����180
			theta.at<int>(i, j) = atan(gradY / gradX) * 57.3;
			theta.at<int>(i, j) = (theta.at<int>(i, j) + 360) % 360;
			gradXY.at<int>(i, j) = sqrt(gradX*gradX + gradY * gradY);
			//XY[i * stepXY + j * (stepXY / step)] = sqrt(gradX * gradX + gradY * gradY);
		}
	}
	/*
	�ھ����������Ҫ��convertScaleAbs()��������ת��ԭ����uint8��ʽ��
	�����޷���ʾͼ�񣬶�ֻ��һ����ɫ�Ĵ��ڡ�
	����ԭ��Ϊ
	void convertScaleAbs(InputArray src, OutputArray dst,
								  double alpha = 1, double beta = 0);
	���п�ѡ����alpha������ϵ����beta�Ǽӵ�����ϵ�һ��ֵ���������uint8���͵�ͼƬ
	���ܣ�ʵ�ֽ�ԭͼƬת��Ϊuint8����
	*/
	convertScaleAbs(imageX, imageX);
	convertScaleAbs(imageY, imageY);
	convertScaleAbs(gradXY, gradXY);
}

/*
�ֲ��Ǽ���ֵ����
���Ÿõ��ݶȷ��򣬱Ƚ�ǰ��������ķ�ֵ��С�����õ����ǰ�����㣬������
���õ�С��ǰ����������һ�㣬����Ϊ0��
imageInput ����õ��ݶ�ͼ��
imageOutput ����ķǼ���ֵ����ͼ��
theta ÿ�����ص���ݶȷ���Ƕ�
imageX X�����ݶ�
imageY Y�����ݶ�
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
			//ִ��continueʱ����ִ�б��ε�����ʣ�ಿ�֣�����ʼ��һ�ε���
			//ִ��breakʱ����ִ�а�������ѭ������ִ����һ�׶�
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

			int direction = theta.at<int>(i, j); //�õ��ݶȵĽǶ�ֵ
			int g1 = 0;
			int g2 = 0;
			int g3 = 0;
			int g4 = 0;
			double tmp1 = 0.0; //���������ص��ֵ�õ��ĻҶ���
			double tmp2 = 0.0;
			//��C�У���abs(),labs(),fabs()�ֱ����int ,long ,double ���͵ľ���ֵ��
			//����C++�У�abs()�ѱ����أ��������ڸ�������
			double weight = fabs((double)imageY.at<uchar>(i, j) / (double)imageX.at<uchar>(i, j));

			//ȥ��Ҳ���ԣ����weight==0����direction==0��180�Ͳ��ò�ֵ��ֱ����ԭ����ֵ
			if (weight == 0)
				weight = 0.0000001;
			/*
			������Щ��ʽ�ĺ���
			https://www.cnblogs.com/love6tao/p/5152020.html �н���
			����g10 * (1 - weight) + g20 * (weight)���������ߵ����ؼ���õ���������
			������w��1-wӦ��д����
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
˫��ֵ�Ļ����ǣ�
ָ��һ������ֵA��һ������ֵB��ѡȡռֱ��ͼ����70%ΪB����BΪ1.5��2����С��A��
�Ҷ�ֵС��A�ģ���Ϊ0,�Ҷ�ֵ����B�ģ���Ϊ255��
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
���Ӵ���:
�Ҷ�ֵ����A��B֮��ģ���������ص��ٽ���8�����Ƿ��лҶ�ֵΪ255�ģ�
��û��255�ģ���ʾ����һ�������ľֲ�����ֵ�㣬�����ų�����Ϊ0��
����255�ģ���ʾ����һ����������Ե�С��������Ŀ���֮�ģ���Ϊ255��
֮���ظ�ִ�иò��裬ֱ�����������һ�����ص㡣

���е���������㷨����ֵΪ255�����ص�����ҵ���Χ����Ҫ��ĵ㣬������Ҫ��ĵ�����Ϊ255��
Ȼ���޸�i,j������ֵ��i,jֵ���л��ˣ��ڸı���i,j�����ϼ���Ѱ��255��Χ����Ҫ��ĵ㡣
����������255�ĵ��޸�����ٰ�����������˵�ľֲ�����ֵ����Ϊ0�����㷨���Լ����Ż�����

����1��imageInput�������������ݶ�ͼ��
����2��lowTh:����ֵ
����3��highTh:����ֵ
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
			//���Ҫ��������ֵ����������ӵ����ϱ�����������߽����ŵ�һ��
			//���ܻ��֮ǰ�����ز���Ӱ�죬������ǰ����ڽ�����ҲҪ�ı�
			//�����ȡ�ķ����ǲ���Ҫ�ı��ĸ����أ�������ǰλ�÷��ص����Ͻ�
			//������һ���ߺ������һ���߲��ܷ��ص����Ͻǣ��ͷֱ����ƺ�����һ��
			//ע��j���ϻ�Ҫ+1��i����+1
			//����Ϊʲô��i>1��j>2����ʵj>1Ҳ���ԣ�ֻ����Ϊj�Ǽ�2����2-2=0��������Ҫ+1����û����
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

	//����ͼƬ
	picture_capture();  

	//����ԭͼ
	Mat srcImage, grayImage;
	srcImage = imread("pic_1.jpg");
	if (!srcImage.data) { cerr << "Failed to load image!\n" << endl; return 1; }

	//��Ϊ�Ҷ�ͼ
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);

	//��˹�˲�
	Mat gausKernel;
	int kernel_size = 5;
	double sigma = 1;
	Gaussian_kernel(kernel_size, sigma, gausKernel);

	Mat gausImage;
	//Convolves an image with kernel �������ں�ʵ�ֶ�ͼ��ľ������
	filter2D(grayImage, gausImage, grayImage.depth(), gausKernel);
	//imshow("gaus image", gausImage);
	//waitKey(0);
	//imwrite("gausImage.jpg", gausImage);

	//����XY�����ݶ�
	Mat imageX, imageY, imageXY;
	//thetaΪ�ݶȷ���theta=arctan(imageY/imageX)
	Mat theta;
	GradDirection(gausImage, imageX, imageY, imageXY, theta);
	//imshow("XY grad", imageXY);
	//waitKey(0);
	//imwrite("GradImage.jpg", imageXY);

	//���ݶȷ�ֵ���зǼ���ֵ����
	Mat localImage;
	NonLocalMaxValue(imageXY, localImage, theta, imageX, imageY);
	//imshow("Non local maxinum image", localImage);
	//waitKey(0);
	//imwrite("localImage.jpg", localImage);

	//˫��ֵ�㷨���ͱ�Ե����
	DoubleThreshold(localImage, 60, 100);
	DoubleThresholdLink(localImage, 60, 100);
	imshow("canny image", localImage);
	waitKey(0);
	imwrite("cannyImage.jpg", localImage);

	return 0;

}


