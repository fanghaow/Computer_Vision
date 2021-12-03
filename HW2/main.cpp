#include <opencv2/opencv.hpp>
#include <string.h>

using namespace std;
using namespace cv;

void findEllipses(string path);
char *filename;

int main(int argc, char **argv)
{
    string filepath;
    if (argc > 1){
        if (argv[1] == "0"){ // debugging
            filepath = "/home/fanghaow/Computer_Vision/HW2/input/ellipses.png";
            filename = "/home/fanghaow/Computer_Vision/HW2/input/ellipses.png";
        }
        else{
            filepath = "/home/fanghaow/Computer_Vision/HW2/input/rice.png";
            filename = "/home/fanghaow/Computer_Vision/HW2/input/rice.png";
        }
    }
    else { // defalut setting
        filepath = "/home/fanghaow/Computer_Vision/HW2/input/rice.png";
        filename = "/home/fanghaow/Computer_Vision/HW2/input/rice.png";
    }

    findEllipses(filepath);

    return 0;
}

void findEllipses(string path)
{
    char* blurImgName = "output/GaussionBlur.jpg";
    char* cannyImgName = "output/canny.jpg";
    char* ellipseImgName = "output/result.jpg";
    Mat img = imread(path, 0);
    CvMemStorage* storage;
    CvSeq* contour;
    // Create dynamic structure and sequence.
    storage = cvCreateMemStorage(0);
    contour = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), storage);
    
    // 1. Origin (Grayscale)
    IplImage *image = cvLoadImage(filename, 0);
    cvShowImage("Origin", image);
    cvWaitKey(0);
    // 2. GaussianBlur
    IplImage *blur_image = cvLoadImage(filename, 0);
    Size ksize = Size(5,5);
    double sigmaX = 2;
    double sigmaY = 2;
    Mat blur_img;
    GaussianBlur(img, blur_img, ksize, sigmaX, sigmaY);
    cv::imwrite(blurImgName, blur_img); // Save blurred result and build a bridge for cpp and c
    blur_image = cvLoadImage(blurImgName, 0);
    cvShowImage("Blurred", blur_image);
    cvWaitKey(0);
    // 3. Canny
    double canny_th1 = 50;
    double canny_th2 = 100;
    IplImage *edge = cvLoadImage(filename, 0);
    cvCanny(blur_image, edge, canny_th1, canny_th2); // 90, 140
    cvSaveImage(cannyImgName, edge);
    cvShowImage("Edge", edge);
    cvWaitKey(0);
    // 4. Find all contours.
    cvFindContours( edge, storage, &contour, sizeof(CvContour),
                    CV_RETR_LIST, CV_CHAIN_APPROX_NONE, cvPoint(0,0));

    // 5 and 6 approximate all the contours and draw ellipses around them
    IplImage *rbgimage = cvLoadImage(filename);
    for (; contour; contour = contour->h_next)
    {
        int pointNum = contour->total; // This is points number in contour
        if (pointNum < 6) // Less than 6 points can not approximate a ellipse!
            continue;

        CvMat* points_f = cvCreateMat(1, pointNum, CV_32FC2);
        CvMat points_i = cvMat(1, pointNum, CV_32SC2, points_f->data.ptr);
        cvCvtSeqToArray(contour, points_f->data.ptr, CV_WHOLE_SEQ);
        cvConvert(&points_i, points_f);

        CvBox2D box = cvFitEllipse2(points_f);
        CvPoint center = cvPointFrom32f(box.center);
        CvSize size;
        size.width = cvRound(box.size.width / 2) + 1;
        size.height = cvRound(box.size.height / 2) + 1;
        double startEngle = 0;
        double endEngle = 360;
        Scalar color = CV_RGB(0,180,0);
        int thickness = 3;

        // cout << size.width * size.height << endl;
        // cout << (float)size.width / size.height << " " << (float)size.height / size.width << endl;
        if (size.width * size.height < 100 || size.width * size.height > 400){
            // cvEllipse(rbgimage, center, size, box.angle, startEngle, endEngle, CV_RGB(20,20,20), thickness);
            continue; // area is not good!
        }
        
        if ((float)size.width / size.height < 0.25 || (float)size.width / size.height > 0.5){
            // cvEllipse(rbgimage, center, size, box.angle, startEngle, endEngle, CV_RGB(180,0,0), thickness);
            continue; // shape ratio is not good!
        }
            
        cvEllipse(rbgimage, center, size, box.angle, startEngle, endEngle, color, thickness); // draw ellipse on the img
    }

    cvSaveImage(ellipseImgName, rbgimage);
    cvShowImage("Result", rbgimage);
    cvWaitKey(0);
}