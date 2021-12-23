#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
int calibrate(int board_w, int board_h, int n_boards);
int bird_view(bool isL, int board_w, int board_h, string intrinsic_file, string input_fname);

int main(int argc, char *argv[])
{
    /* Calibration */
    int boardWidth = 12;
    int boardHeight = 12;
    int boardNum = 23;
    calibrate(boardWidth, boardHeight, boardNum);

    /* BirdsEye */
    string intrinsicFile = "output/intrinsics.xml";
    string birdEyeFname = "input/birdseye/IMG_0";
    int fID = 214;
    for (; fID < 221; fID++) {
        bird_view(false, 12, 12, intrinsicFile, birdEyeFname + to_string(fID) + ".jpg");
    }
    fID = 214;
    for (; fID < 221; fID++) {
        bird_view(true, 12, 12, intrinsicFile, birdEyeFname + to_string(fID) + "L.jpg");
    }

    return 0;
}

int calibrate(int board_w, int board_h, int n_boards)
{
    int board_n = board_w * board_h;
    float image_sf = 0.5f; // image scaling factor
    int delay = 100; // [ms]
    cv::Size board_sz = cv::Size(board_w, board_h);

    string calib_fname = "input/calibration/IMG_0";
    int fileID = 190;

    // ALLOCATE STORAGE
    //
    vector<vector<cv::Point2f> > image_points;
    vector<vector<cv::Point3f> > object_points;

    // Capture corner views: loop until we've got n_boards successful
    // captures (all corners on the board are found).
    //
    double last_captured_timestamp = 0;
    cv::Size image_size;
    while (image_points.size() < (size_t)n_boards) 
    {
        cv::Mat image0, image;
        fileID += 1;
        image0 = cv::imread(calib_fname + to_string(fileID) + ".jpg");
        if (image0.empty()) {
            cout << "[WARN]: Can't open file -> " + calib_fname + to_string(fileID) + ".jpg" << endl;
            n_boards--;
            continue;
        }
        image_size = image0.size();
        cv::resize(image0, image, cv::Size(), image_sf, image_sf, cv::INTER_LINEAR);

        // Find the board
        //
        vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(image, board_sz, corners);

        // Draw it
        //
        drawChessboardCorners(image, board_sz, corners, found);

        // If we got a good board, add it to our data
        //
        double timestamp = static_cast<double>(clock()) / CLOCKS_PER_SEC;
        if (found) { // && timestamp - last_captured_timestamp > 1) {
            last_captured_timestamp = timestamp;
            image ^= cv::Scalar::all(255);
            cv::Mat mcorners(corners);

            // do not copy the data
            mcorners *= (1.0 / image_sf);

            // scale the corner coordinates
            image_points.push_back(corners);
            object_points.push_back(vector<cv::Point3f>());
            vector<cv::Point3f> &opts = object_points.back();

            opts.resize(board_n);
            for (int j = 0; j < board_n; j++) {
                opts[j] = cv::Point3f(static_cast<float>(j / board_w),
                                    static_cast<float>(j % board_w), 0.0f);
            }
            cout << "Collected our " << static_cast<uint>(image_points.size())
                << " of " << n_boards << " needed chessboard images\n" << endl;
        }
        else {
            n_boards--;
        }
        cv::imshow("Calibration", image);
        cv::imwrite("output/calibration/" + to_string(fileID) + "_corners.png", image);

        // show in color if we did collect the image
        if ((cv::waitKey(delay) & 255) == 27)
        return -1;
    }

    // END COLLECTION WHILE LOOP.
    cv::destroyWindow("Calibration");
    cout << "\n\n*** CALIBRATING THE CAMERA...\n" << endl;

    // CALIBRATE THE CAMERA!
    //
    cv::Mat intrinsic_matrix, distortion_coeffs;
    double err = cv::calibrateCamera(
        object_points, image_points, image_size, intrinsic_matrix,
        distortion_coeffs, cv::noArray(), cv::noArray(),
        cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_PRINCIPAL_POINT);

    // SAVE THE INTRINSICS AND DISTORTIONS
    string fpath = "output/intrinsics.xml";
    cout << " *** DONE!\n\nReprojection error is " << err
         << "\nStoring Intrinsics.xml and Distortions.xml files\n\n";
    cv::FileStorage fs(fpath, cv::FileStorage::WRITE);
    fs << "image_width" << image_size.width << "image_height" << image_size.height
       << "camera_matrix" << intrinsic_matrix << "distortion_coefficients"
       << distortion_coeffs;
    fs.release();

    // EXAMPLE OF LOADING THESE MATRICES BACK IN:
    fs.open(fpath, cv::FileStorage::READ);
    cout << "\nimage width: " << static_cast<int>(fs["image_width"]);
    cout << "\nimage height: " << static_cast<int>(fs["image_height"]);
    cv::Mat intrinsic_matrix_loaded, distortion_coeffs_loaded;
    fs["camera_matrix"] >> intrinsic_matrix_loaded;
    fs["distortion_coefficients"] >> distortion_coeffs_loaded;
    cout << "\nintrinsic matrix:" << intrinsic_matrix_loaded;
    cout << "\ndistortion coefficients: " << distortion_coeffs_loaded << endl;

    // Build the undistort map which we will use for all
    // subsequent frames.
    //
    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(intrinsic_matrix_loaded, distortion_coeffs_loaded,
                              cv::Mat(), intrinsic_matrix_loaded, image_size,
                              CV_16SC2, map1, map2);

    // Just run the camera to the screen, now showing the raw and
    // the undistorted image.
    //
    fileID = 190;
    for (;;) {
        cv::Mat image, image0;
        fileID += 1;
        image0 = cv::imread(calib_fname + to_string(fileID) + ".jpg");
        cv::imwrite("output/calibration/" + to_string(fileID) + "_undistorted.png", image0);

        if (image0.empty()) {
            break;
        }
        cv::remap(image0, image, map1, map2, cv::INTER_LINEAR,
            cv::BORDER_CONSTANT, cv::Scalar());
        cv::imshow("Undistorted", image);
        if ((cv::waitKey(delay) & 255) == 27) {
            break;
        }
    }

    return 0;
}

int bird_view(bool isL, int board_w, int board_h, string intrinsic_file, string input_fname)
{
    static int ID = 213;
    ID++;
    // Input Parameters:
    //
    int board_n = board_w * board_h;
    cv::Size board_sz(board_w, board_h);
    cv::FileStorage fs(intrinsic_file, cv::FileStorage::READ);
    cv::Mat intrinsic, distortion;

    fs["camera_matrix"] >> intrinsic;
    fs["distortion_coefficients"] >> distortion;

    if (!fs.isOpened() || intrinsic.empty() || distortion.empty()) {
    cout << "Error: Couldn't load intrinsic parameters from " << intrinsic_file
            << endl;
    return -1;
    }
    fs.release();

    cv::Mat gray_image, image, image0 = cv::imread(input_fname, 1);
    if (image0.empty()) {
    cout << "Error: Couldn't load image " << input_fname << endl;
    return -1;
    }

    // UNDISTORT OUR IMAGE
    //
    cv::undistort(image0, image, intrinsic, distortion, intrinsic);
    cv::cvtColor(image, gray_image, cv::COLOR_BGRA2GRAY);

    // GET THE CHECKERBOARD ON THE PLANE
    //
    vector<cv::Point2f> corners;
    bool found = cv::findChessboardCorners( // True if found
        image,                              // Input image
        board_sz,                           // Pattern size
        corners,                            // Results
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);
    if (!found) {
        cout << "Couldn't acquire checkerboard on " << input_fname << ", only found "
                << corners.size() << " of " << board_n << " corners\n";
        return -1;
    }

    // Get Subpixel accuracy on those corners
    //
    cv::cornerSubPix(
        gray_image,       // Input image
        corners,          // Initial guesses, also output
        cv::Size(11, 11), // Search window size
        cv::Size(-1, -1), // Zero zone (in this case, don't use)
        cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30,
                    0.1));

    // GET THE IMAGE AND OBJECT POINTS:
    // Object points are at (r,c):
    // (0,0), (board_w-1,0), (0,board_h-1), (board_w-1,board_h-1)
    // That means corners are at: corners[r*board_w + c]
    //
    cv::Point2f objPts[4], imgPts[4];
    objPts[0].x = 0;
    objPts[0].y = 0;
    objPts[1].x = board_w - 1;
    objPts[1].y = 0;
    objPts[2].x = 0;
    objPts[2].y = board_h - 1;
    objPts[3].x = board_w - 1;
    objPts[3].y = board_h - 1;
    imgPts[0] = corners[0];
    imgPts[1] = corners[board_w - 1];
    imgPts[2] = corners[(board_h - 1) * board_w];
    imgPts[3] = corners[(board_h - 1) * board_w + board_w - 1];

    // DRAW THE POINTS in order: B,G,R,YELLOW
    //
    cv::circle(image, imgPts[0], 9, cv::Scalar(255, 0, 0), 3);
    cv::circle(image, imgPts[1], 9, cv::Scalar(0, 255, 0), 3);
    cv::circle(image, imgPts[2], 9, cv::Scalar(0, 0, 255), 3);
    cv::circle(image, imgPts[3], 9, cv::Scalar(0, 255, 255), 3);                                         

    // DRAW THE FOUND CHECKERBOARD
    //
    cv::drawChessboardCorners(image, board_sz, corners, found);
    cv::imshow("Checkers", image);
    if (isL) {
        cv::imwrite("output/birdseye/" + to_string(ID) + "L_checkerboard.png", image);
    } else {
        cv::imwrite("output/birdseye/" + to_string(ID) + "_checkerboard.png", image);
    }

    // FIND THE HOMOGRAPHY
    //
    cv::Mat H = cv::getPerspectiveTransform(objPts, imgPts);

    // LET THE USER ADJUST THE Z HEIGHT OF THE VIEW
    //
    cout << "\nPress 'd' for lower birdseye view, and 'u' for higher (it adjusts the apparent 'Z' height), Esc to exit" << endl;
    double Z = 15;
    cv::Mat birds_image;
    for (;;) 
    {
        // escape key stops
        H.at<double>(2, 2) = Z;
        // USE HOMOGRAPHY TO REMAP THE VIEW
        //
        cv::warpPerspective(image,			// Source image
                            birds_image, 	// Output image
                            H,              // Transformation matrix
                            image.size(),   // Size for output image
                            cv::WARP_INVERSE_MAP | cv::INTER_LINEAR,
                            cv::BORDER_CONSTANT, cv::Scalar::all(0) // Fill border with black
                            );
        cv::imshow("Birds_Eye", birds_image);
        if (isL) {
            cv::imwrite("output/birdseye/" + to_string(ID) + "L_birdseye.png", birds_image);
        } else {
            cv::imwrite("output/birdseye/" + to_string(ID) + "_birdseye.png", birds_image);
        }
        int key = cv::waitKey() & 255;
        if (key == 'u')
            Z += 0.5;
        if (key == 'd')
            Z -= 0.5;
        if (key == 27)
            break;
    }

    // SHOW ROTATION AND TRANSLATION VECTORS
    //
    vector<cv::Point2f> image_points;
    vector<cv::Point3f> object_points;
    for (int i = 0; i < 4; ++i) {
        image_points.push_back(imgPts[i]);
        object_points.push_back(cv::Point3f(objPts[i].x, objPts[i].y, 0));
    }
    cv::Mat rvec, tvec, rmat;
    cv::solvePnP(object_points, 	// 3-d points in object coordinate
                image_points,  	// 2-d points in image coordinates
                intrinsic,     	// Our camera matrix
                cv::Mat(),     	// Since we corrected distortion in the
                                // beginning,now we have zero distortion
                                // coefficients
                rvec, 			// Output rotation *vector*.
                tvec  			// Output translation vector.
                );
    cv::Rodrigues(rvec, rmat);

    // PRINT AND EXIT
    cout << "rotation matrix: " << rmat << endl;
    cout << "translation vector: " << tvec << endl;
    cout << "homography matrix: " << H << endl;
    cout << "inverted homography matrix: " << H.inv() << endl;

    return 1;
}