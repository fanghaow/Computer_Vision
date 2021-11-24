#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <string>

using namespace std;
#define FADE_IN 1
#define FADE_OUT -1
#define ZOOM_IN 1
#define ZOOM_OUT -1
int COL = 1200;
int ROW = 800;

string TextCont("ID:3180100675");
cv::Point TextPos(COL * 0.4, ROW * 0.7);
int TextFont = cv::FONT_HERSHEY_DUPLEX;
double TextWidth = 3.0;
cv::Scalar TextColor(CV_RGB(118, 185, 0));

void fade(cv::Mat src, int flag); /* Fade in and out */
void slide(cv::Mat src); /* Slide in and out */
void zoom(cv::Mat src, int flag); /* Zoom in and out */
void erase(cv::Mat src1, cv::Mat src2); /* Erase src1, src2 comes out */
void affine(cv::Mat src);

int main()
{
    cv::Size showSize = cv::Size(COL, ROW);

    cv::Mat img1, img2;
    cv::Mat resizedImg1, resizedImg2;
    cv::Mat src;
    img1 = cv::imread("input/night.jpg");
    img2 = cv::imread("input/sky.jpg");
    if (img1.empty())
        cout << "[ERROR]: opening image file failed" << endl;
    else{
        cv::resize(img1, resizedImg1, showSize, 0, 0, cv::INTER_LINEAR);
        cv::resize(img2, resizedImg2, showSize, 0, 0, cv::INTER_LINEAR);

        fade(resizedImg1, 1); // Fade in
        fade(resizedImg1, -1); // Fade out
        img1 = cv::imread("input/pink.jpg");
        cv::resize(img1, resizedImg1, showSize, 0, 0, cv::INTER_LINEAR);
        slide(resizedImg1);
        img1 = cv::imread("input/sun.jpg");
        cv::resize(img1, resizedImg1, showSize, 0, 0, cv::INTER_LINEAR);
        zoom(resizedImg1, ZOOM_IN);
        img1 = cv::imread("input/cat.jpg");
        cv::resize(img1, resizedImg1, showSize, 0, 0, cv::INTER_LINEAR);
        erase(resizedImg1, resizedImg2);
        img1 = cv::imread("input/dog.jpg");
        cv::resize(img1, resizedImg1, showSize, 0, 0, cv::INTER_LINEAR);
        affine(resizedImg1);
    }

    /* Play the video */
    cv::VideoCapture cap("input/stars.mp4");
    if (!cap.isOpened()){
        cout << "[ERROR]: openning video file failed" << endl;
    }

    while (true)
    {
        cv::Mat frame;
        cv::Mat resizedframe;
        cap >> frame;

        if (frame.empty())
            break;

        cv::resize(frame, resizedframe, showSize, 0, 0, cv::INTER_LINEAR);
        cv::putText(resizedframe, TextCont, TextPos, TextFont, TextWidth, TextColor);
        cv::imshow("stars", resizedframe);
        char c = (char)cv::waitKey(1);
        if (c == 27)
            break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}

void fade(cv::Mat src, int flag)
{
    cv::Mat fadeImg;
    int step_size = 5;
    int step = 255 / step_size;
    for (int i = 0; i < step; i++)
    {
        if (flag == FADE_IN)
            cv::subtract(src, (255 - i * step_size), fadeImg);
        else if (flag == FADE_OUT)
            cv::subtract(src, (i * step_size), fadeImg);
        cv::putText(fadeImg, TextCont, TextPos, TextFont, TextWidth, TextColor);
        cv::imshow("fadeImg", fadeImg);
        cv::waitKey(15);
    }
    cv::destroyAllWindows();
}

void slide(cv::Mat src) // slide from left to right
{
    cv::Mat slideImg = cv::Mat::zeros(src.size(), src.type());
    int col1, col2, row1, row2;
    col1 = col2 = row1 = row2 = 0;
    int step = 100;
    int stepR = ROW / step;
    int stepC = COL / step;
    for (int i = 0; i < step; i++)
    {
        col2 += stepC;
        row2 += stepR;
        cv::Mat roiImg = src(cv::Rect(col1,row1,col2,row2));
        roiImg.copyTo(slideImg(cv::Rect(col1,row1,col2,row2)));
        cv::putText(slideImg, TextCont, TextPos, TextFont, TextWidth, TextColor);
        cv::imshow("Slide", slideImg);
        cv::waitKey(15);
    }
    cv::destroyAllWindows();
}

void zoom(cv::Mat src, int flag)
{
    cv::Mat zoomedImg = cv::Mat::zeros(src.size(), src.type());
    cv::Mat scaledImg;

    int step = 100;
    for (int i = 0; i < step; i++)
    {    
        float scale = (float)(i + 1) / step;
        // cout << "scale = " << scale << endl;
        cv::Size scaledSize = cv::Size(COL * scale, ROW * scale); 
        cv::resize(src, scaledImg, scaledSize, 0, 0, cv::INTER_LINEAR);
        
        cv::Rect roi(0, 0, COL * scale, ROW * scale);
        scaledImg.copyTo(zoomedImg(roi));
        
        cv::putText(zoomedImg, TextCont, TextPos, TextFont, TextWidth, TextColor);
        cv::imshow("ZOOM", zoomedImg);
        cv::waitKey(15);
    }
    cv::destroyAllWindows();
}

void erase(cv::Mat src1, cv::Mat src2)
{
    int step = 100;
    for (int i = 0; i < step; i++)
    {    
        int bound = i * COL / step;
        cv::Mat erasedImg = src1.clone();
        // cv::Rect roi1(bound,0,COL,ROW);
        cv::Rect roi2(0,0,bound,ROW);
        src2(roi2).copyTo(erasedImg(roi2));
        cv::putText(erasedImg, TextCont, TextPos, TextFont, TextWidth, TextColor);
        cv::imshow("Erase", erasedImg);
        cv::waitKey(15);
    }
    cv::destroyAllWindows();
}

void affine(cv::Mat src)
{
    int step = 100;
    for (int i = 0; i < step; i++)
    {    
        float x = (float)i / step;
        cv::Point2f srcTri[3];
        srcTri[0] = cv::Point2f(0.f, 0.f);
        srcTri[1] = cv::Point2f(src.cols - 1.f, 0.f);
        srcTri[2] = cv::Point2f(0.f, src.rows - 1.f);
        cv::Point2f dstTri[3];
        dstTri[0] = cv::Point2f(0.f, src.rows*x);
        dstTri[2] = cv::Point2f(src.cols*(1.f-x), src.rows*(1.f-x));
        dstTri[1] = cv::Point2f(src.cols*x, src.rows*x);
        cv::Mat warp_mat = cv::getAffineTransform(srcTri, dstTri);
        cv::Mat warp_dst = cv::Mat::zeros(src.rows, src.cols, src.type());
        cv::warpAffine(src, warp_dst, warp_mat, warp_dst.size());

        cv::Point center = cv::Point(warp_dst.cols/2, warp_dst.rows/2);
        double angle = 180;
        double scale = 1.0;
        cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);
        cv::Mat warp_rotate_dst;
        cv::warpAffine(warp_dst, warp_rotate_dst, rot_mat, warp_dst.size());

        cv::putText(warp_rotate_dst, TextCont, TextPos, TextFont, TextWidth, TextColor);
        cv::imshow("Affine", warp_rotate_dst);
        cv::waitKey(15);
    }
    cv::destroyAllWindows();
}