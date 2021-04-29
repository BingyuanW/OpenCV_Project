#include <iostream> // for standard I/O

#include <opencv2/core.hpp>     // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui.hpp>  // OpenCV window I/O
using namespace std;
using namespace cv;



int main(int argc, char** argv)

{
    int cannyLowThreshold = 9 ;
    Mat I ;
    I = imread("./bus.jpg",1);

    cvtColor(I, I, COLOR_BGR2GRAY);

    GaussianBlur(I,I,Size(11, 11), 0,0);
    Canny(I, I,cannyLowThreshold, cannyLowThreshold*3, 3);


    namedWindow(" image", WINDOW_NORMAL) ;
    imshow(" image", 255-I); 
       
    cv::waitKey(0);


    //return 0 ;
}
// ![get-mssim]





