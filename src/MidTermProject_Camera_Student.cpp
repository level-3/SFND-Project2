/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
//#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    bool verbose = true;
    bool visualize_result = true;
    bool visualize_matches = true;
    bool print_header = true;
    bool print_to_file = true;
    int match_count ;
    double t_extraction, t_match;
    int num_errors = 0;

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    const char *descriptor[5] = {"BRIEF", "ORB", "FREAK", "SIFT", "AKAZE"};
    const char *detector[7] = {"SHITOMASI","HARRIS", "FAST", "BRISK", "ORB", "SIFT", "AKAZE"};

    ofstream myfile;
    if(print_to_file)
    {
        string filename = "../src/results/data" ; 
        myfile.open(filename + ".csv");        
    }


    for(auto it_detector = std::begin(detector); it_detector != std::end(detector); ++it_detector)
    {

    for(auto it_descriptor = std::begin(descriptor); it_descriptor != std::end(descriptor); ++it_descriptor)
    {

        //https://knowledge.udacity.com/questions/105392
        if( (*it_detector != "AKAZE") && (*it_descriptor =="AKAZE") )
        {
            continue;   //AKAZE descriptors worked only with AKAZE detectors
        } 
        if( (*it_detector == "SIFT") && (*it_descriptor =="ORB") )
        {
            continue;   //known error - SIFT descriptors are not extracted with ORB detectors
        } 

    try {


    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;

        if (dataBuffer.size() == dataBufferSize)
        {
            dataBuffer.erase(dataBuffer.begin());
        }

        dataBuffer.push_back(frame);

        //// EOF STUDENT ASSIGNMENT
        if (verbose)
            cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        string detectorType = *it_detector; //SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
        double t_detect = (double)cv::getTickCount();  


        if (detectorType.compare("SHITOMASI") == 0)
            {
                detKeypointsShiTomasi(keypoints, imgGray, visualize_result);
            }
        else if (detectorType.compare("HARRIS") == 0)
            {
                detKeypointsHarris(keypoints, imgGray, visualize_result);
            }
        else
            {
                detKeypointsModern(keypoints, imgGray, detectorType, visualize_result);
            }

        t_detect = ((double)cv::getTickCount() - t_detect) / cv::getTickFrequency();
        if (verbose)
            cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t_detect / 1.0 << " ms" << endl;
        
        int detected_keypoints = keypoints.size();

        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        float x_min, x_max, y_min, y_max;
        x_min = vehicleRect.x;
        x_max = x_min + vehicleRect.width;
        y_min = vehicleRect.y;
        y_max = y_min + vehicleRect.height;

        if (bFocusOnVehicle)
        {
            if (verbose)
                cout << "before KEYPOINTS:\t" << keypoints.size() << endl;

            for (auto it = keypoints.begin(); it != keypoints.end();)
                {
                    cv::KeyPoint kp = *it;

                    if ((kp.pt.x >= x_min && kp.pt.x <= x_max) && (kp.pt.y >= y_min && kp.pt.y <= y_max))
                        {
                            if (verbose)
                                cout << "X:\t" << kp.pt.x << "\t\tY:\t" << kp.pt.y << endl;
                            it++;
                        }
                    else
                        {
                            it = keypoints.erase(it);
                        }
                }

            if (verbose)
                cout << "after KEYPOINTS:\t" << keypoints.size() << endl;
        }
        detected_keypoints = keypoints.size();
        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;

        if (verbose)
            cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        string descriptorType = *it_descriptor ; // descriptorT;"ORB" ;// // BRIEF, ORB, FREAK, AKAZE, SIFT


        t_extraction = descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
        
        
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        if (verbose)
            cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";     // MAT_BF, MAT_FLANN
            string descriptorTypeClass = "DES_HOG"; // DES_BINARY, DES_HOG
            string selectorType = "SEL_KNN";      // SEL_NN, SEL_KNN

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            if (descriptorType.compare("AKAZE") == 0 || descriptorType.compare("BRIEF") == 0 || descriptorType.compare("BRISK") == 0 || descriptorType.compare("ORB") == 0)
                {   
                    descriptorTypeClass = "DES_BINARY";
                }
            
            t_match = matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                            (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                            matches, descriptorTypeClass, matcherType, selectorType);


            match_count = matches.size();

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            if (verbose)
                cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                if (visualize_matches)
                {
                    cout << "Press key to continue to next image" << endl;
                    cv::waitKey(0); // wait for key to be pressed
                }
            }
            bVis = false;
        }

        if (print_header)
        {            
            myfile << "detect T" <<
                    "," << "t_detect" <<
                    "," << "keypoints" <<
                    "," << "descr T" <<
                    "," << "t_extract" <<
                    "," << "matches" <<
                    "," << "t_match" <<
                    endl;
            

            print_header = false;            
        }

        if (print_to_file)
        {
            myfile  << detectorType <<
                    "," << 1000 * t_detect / 1.0 << 
                    "," << detected_keypoints << 
                    "," << descriptorType << 
                    "," << 1000 * t_extraction / 1.0 << 
                    "," << match_count << 
                    "," << 1000 * t_match / 1.0 << 
                    endl;
        }

        

    } // eof loop over all images

    dataBuffer.clear();
    
    }
    catch(cv::Exception e) {
        cerr << "Detector: " <<  *it_detector << "\tDescriptor: " << *it_descriptor  << endl;
        cerr << "Error:" << e.what() <<  endl;
        num_errors++;
    }   


    }
    }    


    if(print_to_file)
        myfile.close();

    cout << "Errors : " << num_errors << endl;

    return 0;
}