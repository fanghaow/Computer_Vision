#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <string>

using namespace std;

#define FADE_IN 1
#define FADE_OUT -1
#define ZOOM_IN 1
#define ZOOM_OUT -1

/* Global variables */
int COL = 1200;
int ROW = 800;
int delay = 15;
const string outputName = "output/stars.mp4";
static cv::VideoWriter outputVideo;
cv::Size showSize = cv::Size(COL, ROW);

/* Global variables for cv::putText */
const string TextCont("ID:3180100675");
cv::Point TextPos(COL * 0.35, ROW * 0.95);
int TextFont = cv::FONT_HERSHEY_TRIPLEX;
double TextWidth = 3.0;
cv::Scalar TextColor(CV_RGB(255, 255, 0));

void fade(cv::Mat src, int flag); /* Fade in and out */
void slide(cv::Mat src); /* Slide */
void zoom(cv::Mat src, int flag); /* Zoom in and out */
void erase(cv::Mat src1, cv::Mat src2); /* Erase src1, src2 comes out */
void affine(cv::Mat src); /* Afine */

int main(int argc, char **argv)
{
    string path;
    if (argc > 1)
        path = argv[1];
    else
        path = "input/";

    /* Video Input and Output init */
    cv::VideoCapture cap(path + "stars.mp4");
    int ex = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
    outputVideo.open(outputName, ex, cap.get(cv::CAP_PROP_FPS), showSize, true);
    if (!outputVideo.isOpened())
    {
        cout << "Could not open the output video to write: " << outputName << endl;
        return -1;
    }

    /* Pictures Input and Process */
    cv::Mat img1, img2, resizedImg1, resizedImg2;
    img1 = cv::imread(path + "sea.jpg");
    img2 = cv::imread(path + "mountain.jpg");
    if (img1.empty() || img2.empty())
        cout << "[ERROR]: opening image file failed" << endl;
    else{
        // /* 1. Fade */
        cv::resize(img1, resizedImg1, showSize, 0, 0, cv::INTER_LINEAR);
        cv::resize(img2, resizedImg2, showSize, 0, 0, cv::INTER_LINEAR);
        fade(resizedImg1, FADE_IN); // Fade in
        fade(resizedImg1, FADE_OUT); // Fade out
        /* 2. Slide */
        img1 = cv::imread(path + "lake.jpg");
        cv::resize(img1, resizedImg1, showSize, 0, 0, cv::INTER_LINEAR);
        slide(resizedImg1);
        /* 3. Zoom */
        img1 = cv::imread(path + "grassland.jpg");
        cv::resize(img1, resizedImg1, showSize, 0, 0, cv::INTER_LINEAR);
        zoom(resizedImg1, ZOOM_IN); // Zoom in
        zoom(resizedImg1, ZOOM_OUT); // Zoom out
        /* 4. Erase */
        img1 = cv::imread(path + "snow.jpg");
        cv::resize(img1, resizedImg1, showSize, 0, 0, cv::INTER_LINEAR);
        erase(resizedImg1, resizedImg2);
        /* 5. Affine */
        img1 = cv::imread(path + "dusk.jpg");
        cv::resize(img1, resizedImg1, showSize, 0, 0, cv::INTER_LINEAR);
        affine(resizedImg1);
    }

    /* Play and Save the video */
    if (!cap.isOpened()){
        cout << "[ERROR]: openning video file failed" << endl;
    }
    while (true)
    {
        cv::Mat frame;
        cv::Mat resizedframe;
        cap >> frame; // read in one frame
        if (frame.empty())
            break;
        cv::resize(frame, resizedframe, showSize, 0, 0, cv::INTER_LINEAR);
        cv::putText(resizedframe, TextCont, TextPos, TextFont, TextWidth, TextColor);
        cv::imshow("stars", resizedframe);
        char c = (char)cv::waitKey(1);
        outputVideo << resizedframe;
        if (c == 27) // [esc]
            break;
    }

    /* Release all */
    cap.release();
    cv::destroyAllWindows();
    cout << "Success!" << endl;

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
        cv::waitKey(2 * delay);
        outputVideo << fadeImg;
    }
    cv::destroyAllWindows();
}

void slide(cv::Mat src) // slide from top to bottom
{
    cv::Mat slideImg = cv::Mat::zeros(src.size(), src.type());
    int col1, col2, row1, row2;
    col1 = col2 = row1 = row2 = 0;
    int step = 100;
    int stepR = ROW / step;
    // int stepC = COL / step;
    for (int i = 0; i < step; i++)
    {
        // col2 += stepC;
        col2 = COL;
        row2 += stepR;
        cv::Mat roiImg = src(cv::Rect(col1,row1,col2,row2));
        roiImg.copyTo(slideImg(cv::Rect(col1,row1,col2,row2)));
        cv::putText(slideImg, TextCont, TextPos, TextFont, TextWidth, TextColor);
        cv::imshow("Slide", slideImg);
        cv::waitKey(delay);
        outputVideo << slideImg;
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
        float scale;
        if (flag == ZOOM_IN)
            scale = (float)(i + 1) / step;
        else if (flag == ZOOM_OUT)
            scale = (float)(step - i) / step;
        cv::Size scaledSize = cv::Size(COL * scale, ROW * scale); 
        cv::resize(src, scaledImg, scaledSize, 0, 0, cv::INTER_LINEAR);
        cv::Rect roi(0, 0, COL * scale, ROW * scale);
        roi = roi + cv::Point( (COL - COL * scale) / 2, (ROW - ROW * scale) / 2);
        scaledImg.copyTo(zoomedImg(roi));
        cv::putText(zoomedImg, TextCont, TextPos, TextFont, TextWidth, TextColor);
        cv::imshow("ZOOM", zoomedImg);
        cv::waitKey(delay);
        outputVideo << zoomedImg;
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
        cv::Rect roi(0,0,bound,ROW);
        src2(roi).copyTo(erasedImg(roi));
        cv::putText(erasedImg, TextCont, TextPos, TextFont, TextWidth, TextColor);
        cv::imshow("Erase", erasedImg);
        cv::waitKey(delay);
        outputVideo << erasedImg;
    }
    cv::destroyAllWindows();
}

void affine(cv::Mat src)
{
    int step = 100;
    for (int i = 0; i < step; i++)
    {    
        float x = (float)i / step;
        /* Affine */
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
        /* Rotate */
        cv::Point center = cv::Point(warp_dst.cols/2, warp_dst.rows/2);
        double angle = 180;
        double scale = 1.0;
        cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);
        cv::Mat warp_rotate_dst;
        cv::warpAffine(warp_dst, warp_rotate_dst, rot_mat, warp_dst.size());
        /* Put texts on fig and save it into the new video */
        cv::putText(warp_rotate_dst, TextCont, TextPos, TextFont, TextWidth, TextColor);
        cv::imshow("Affine", warp_rotate_dst);
        cv::waitKey(delay);
        outputVideo << warp_rotate_dst;
    }
    cv::destroyAllWindows();
}