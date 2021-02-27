/*
https://docs.opencv.org/master/d3/dc1/tutorial_basic_linear_transform.html
*/

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

// we're NOT "using namespace std;" here, to avoid collisions between the beta variable and std::beta in c++17
using std::cin;
using std::cout;
using std::endl;
using namespace cv;

Mat image, new_image ;

int alpha = 1; /*< Simple contrast control */
int beta = 0;       /*< Simple brightness control */

static void ContrastBrightness(int, void*){
    
    for( int y = 0; y < image.rows; y++ ) {
        for( int x = 0; x < image.cols; x++ ) {
            for( int c = 0; c < image.channels(); c++ ) {
                new_image.at<Vec3b>(y,x)[c] =
                saturate_cast<uchar>( alpha*image.at<Vec3b>(y,x)[c] + beta );
            }
        }
    }   
    //image.convertTo(new_image, -1, alpha, beta);
    imshow("Original Image", image);
    imshow("New Image", new_image);

}

int main( int argc, char** argv )
{
    
    CommandLineParser parser( argc, argv, "{@input | lena.jpg | input image}" );
    image = imread( samples::findFile( parser.get<String>( "@input" ) ) );
    if( image.empty() )
    {
      cout << "Could not open or find the image!\n" << endl;
      cout << "Usage: " << argv[0] << " <Input image>" << endl;
      return -1;
    }
    
    //image = imread( "lena.jpg" );
    new_image = Mat::zeros( image.size(), image.type() );

    cout << " Basic Linear Transforms " << endl;
    cout << "-------------------------" << endl;
    //cout << "* Enter the alpha value [1.0-3.0]: "; cin >> alpha;
    //cout << "* Enter the beta value [0-100]: ";    cin >> beta;

    //线性矫正和Gamma矫正的比较
    Mat basicImage = imread("basic.png");
    //线性矫正
    Mat linearImage = Mat::zeros( basicImage.size(), basicImage.type() );
    basicImage.convertTo(linearImage, -1, 1.3, 40); //alpha=1.3,beta=40
    //Gamma矫正
    Mat gammaImage = Mat::zeros( basicImage.size(), basicImage.type() );
    Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for( int i = 0; i < 256; ++i)
        p[i] = saturate_cast<uchar>(pow(i / 255.0, 0.4) * 255.0); //gamma=0.4
    LUT(basicImage, lookUpTable, gammaImage);
    //三张图片水平连接
    hconcat(basicImage, linearImage, linearImage);
    hconcat(linearImage, gammaImage, gammaImage);
 
    imshow("Image Correction", gammaImage) ;

    //添加两条Trackbar
    namedWindow("New Image", WINDOW_AUTOSIZE); // Create Window
    createTrackbar("Contrast", "New Image", &alpha, 3, ContrastBrightness);
    createTrackbar("Brightness", "New Image", &beta, 100, ContrastBrightness);
    ContrastBrightness(0,0);
    waitKey(0);

    return 0;
}