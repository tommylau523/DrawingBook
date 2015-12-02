//
//  ViewController.m
//  DrawingBook
//
//  Created by Tommy Lau on 11/17/15.
//  Copyright © 2015 Tommy Lau. All rights reserved.
//

#import "ViewController.h"
#include <stdlib.h>
#include <iostream>
#ifdef __cplusplus
#include <opencv2/opencv.hpp>
//#include "opencv2/highgui/ios.h"
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/nonfree/nonfree.hpp"
//#include "armadillo"
#endif

// Include stdlib.h and std namespace so we can mix C++ code in here

using namespace std;

@interface ViewController (){
    UIImageView *liveView_; // Live output from the camera
    CvVideoCamera *videoCamera_; // OpenCV wrapper class to simplfy camera access through AVFoundation;
    cv::vector<cv::KeyPoint> template_keypoints;
    cv::Mat template_im;
    cv::Mat template_descriptor;
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> extractor;
    cv::Ptr<cv::BFMatcher> matcher;
    cv::Mat intrinsics;
}
@property (nonatomic, retain) CvVideoCamera* videoCamera;
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    UIImage *template_image = [UIImage imageNamed:@"template.jpg"];
    detector = new cv::OrbFeatureDetector(3000);
    extractor = new cv::OrbDescriptorExtractor;
    matcher = new cv::BFMatcher(cv::NORM_HAMMING, true);
    template_im = [self cvMatFromUIImage:template_image];
    template_keypoints = getKeyPoints(template_im, detector);
    extractor->compute(template_im, template_keypoints, template_descriptor);
    intrinsics = cv::Mat::zeros(3,3,CV_64F);
    intrinsics.at<double>(0,0) = 3043.72;
    intrinsics.at<double>(1,1) = 3043.72;
    intrinsics.at<double>(2,2) = 1;
    intrinsics.at<double>(0,2) = 1196;
    intrinsics.at<double>(1,2) = 1604;
    
    // 1. Setup the your OpenCV view, so it takes up the entire App screen......
    int view_width = self.view.frame.size.width;
    int view_height = (640*view_width)/480; // Work out the viw-height assuming 640x480 input
    int view_offset = (self.view.frame.size.height - view_height)/2;
    liveView_ = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, view_offset, view_width, view_height)];
    [self.view addSubview:liveView_];
    
    
    // 4. Initialize the camera parameters and start the camera (inside the App)
    videoCamera_ = [[CvVideoCamera alloc] initWithParentView:liveView_];
    videoCamera_.delegate = self;
    videoCamera_.defaultFPS = 60;
    videoCamera_.grayscaleMode = NO;
    videoCamera_.rotateVideo = YES;
    
    // This chooses whether we use the front or rear facing camera
    videoCamera_.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
    
    // This is used to set the image resolution
    videoCamera_.defaultAVCaptureSessionPreset = AVCaptureSessionPreset640x480;
    
    // This is used to determine the device orientation
    videoCamera_.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationLandscapeLeft;
    
    // This starts the camera capture
    [videoCamera_ start];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}



- (void)processImage:(cv::Mat&) image;
{
    
    // Do some OpenCV stuff with the image
    cv::Mat image_copy, gray, template_gray;
    cv::Mat image_descriptor;
    std::vector<cv::DMatch> matches;
    
    cvtColor(image, image_copy, CV_BGRA2BGR);
    cvtColor(image_copy, gray, CV_BGRA2GRAY);
    cvtColor(template_im, template_gray, CV_BGRA2GRAY);
    
    @try{
        cv::vector<cv::KeyPoint> image_keypoints = getKeyPoints(image_copy, detector);
        extractor->compute(image_copy, image_keypoints, image_descriptor);
        if(image_descriptor.cols < 20){
            return;
        }
        matcher->match(template_descriptor, image_descriptor, matches);

        
        double max_dist = 0; double min_dist = 300;
        
        //-- Quick calculation of max and min distances between keypoints
        for( int i = 0; i < matches.size(); i++ )
        { double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }
        
        std::vector< cv::DMatch > good_matches;
        
        for( int i = 0; i < matches.size(); i++ )
        { if( matches[i].distance < 3*min_dist )
        { good_matches.push_back( matches[i]); }
        }
        
        
        cv::vector<cv::Point3f> source;
        cv::vector<cv::Point2f> dest;
        
        for(int i = 0; i < good_matches.size(); i++){
            source.push_back(cv::Point3f(template_keypoints[good_matches[i].queryIdx].pt.x,
                                         template_keypoints[good_matches[i].queryIdx].pt.y,
                                         0));
            dest.push_back(image_keypoints[good_matches[i].trainIdx].pt);
        }
        if(good_matches.size() >= 10){
            //cv::Mat H = cv::findHomography(source, dest, CV_RANSAC);
            //cv::Mat E = get_extrinsics(intrinsics, H);
            //cv::Mat camera = intrinsics*E;
            cv::Mat rvec, tvec;
            cv::Mat distCoeffs(4,1,cv::DataType<double>::type);
            distCoeffs.at<double>(0) = 0;
            distCoeffs.at<double>(1) = 0;
            distCoeffs.at<double>(2) = 0;
            distCoeffs.at<double>(3) = 0;
            cv::solvePnPRansac(source, dest, intrinsics, distCoeffs, rvec, tvec,false, 200, 8.0, good_matches.size()/2);
            
            
            std::vector<cv::Point3f> proj_corners(4);
            std::vector<cv::Point2f> scene_proj_corners(4);
            proj_corners[0] = cv::Point3f(0,0,0);
            proj_corners[1] = cv::Point3f( template_im.cols, 0, 0 );
            proj_corners[2] = cv::Point3f( template_im.cols, template_im.rows, 0 );
            proj_corners[3] = cv::Point3f( 0, template_im.rows, 0 );
            //cv::perspectiveTransform(proj_corners, scene_proj_corners, camera);
            cv::projectPoints(proj_corners, rvec, tvec, intrinsics, distCoeffs, scene_proj_corners);
            
            
            cv::line( image_copy, scene_proj_corners[0], scene_proj_corners[1], cv::Scalar(255, 255, 0), 4 );
            cv::line( image_copy, scene_proj_corners[1], scene_proj_corners[2], cv::Scalar( 255, 255, 0), 4 );
            cv::line( image_copy, scene_proj_corners[2], scene_proj_corners[3], cv::Scalar( 255, 255, 0), 4 );
            cv::line( image_copy, scene_proj_corners[3], scene_proj_corners[0], cv::Scalar( 255, 255, 0), 4 );
            
            
//            std::vector<cv::Point2f> obj_corners(4);
//            obj_corners[0] = cvPoint(0,0);
//            obj_corners[1] = cvPoint( template_im.cols, 0 );
//            obj_corners[2] = cvPoint( template_im.cols, template_im.rows );
//            obj_corners[3] = cvPoint( 0, template_im.rows );
//            std::vector<cv::Point2f> scene_corners(4);
            //cv::perspectiveTransform( obj_corners, scene_corners, H);
            //cout << H << endl;
            //cout << scene_corners << endl;
            //-- Draw lines between the corners (the mapped object in the scene - image_2 )
            //cv::line( image_copy, scene_corners[0], scene_corners[1], cv::Scalar(0, 255, 0), 4 );
            //cv::line( image_copy, scene_corners[1], scene_corners[2], cv::Scalar( 0, 255, 0), 4 );
            //cv::line( image_copy, scene_corners[2], scene_corners[3], cv::Scalar( 0, 255, 0), 4 );
            //cv::line( image_copy, scene_corners[3], scene_corners[0], cv::Scalar( 0, 255, 0), 4 );
            
        }
    } @catch(NSException *theException) {
        cout << theException.name << endl;
        cout << theException.reason << endl;
    }
    
    //cv::drawMatches(gray, K, template_gray, K_template, good_matches, image);
    cvtColor(image_copy, image, CV_BGR2BGRA);
    
}

cv::Mat get_extrinsics(cv::Mat intrinsics, cv::Mat H){
    cv::Mat U, l, V, Hp, t, Omega, o;
    cout << intrinsics << endl;
    Hp = intrinsics.inv() * H;
    t = Hp.col(2);
    Hp = Hp.colRange(0, 2);
    
    cv::SVD::compute(Hp, l, U, V, cv::SVD::FULL_UV);
    
    cv::Mat L = cv::Mat::eye(3, 2, CV_64F);
    Omega = U*L*V.t();
    o = Omega.col(0).cross(Omega.col(1));
    cv::hconcat(Omega, o, Omega);
    
    Omega.col(2) *= cv::determinant(Omega.t()*Omega);
    float scale = 0;
    for(int m = 0; m < 3; m++){
        for(int n = 0; n < 2; n++){
            scale += Hp.at<double>(n,m) / Omega.at<double>(n,m);
        }
    }
    scale /= 6.0;
    t /= scale;
    cv::hconcat(Omega, t, Omega);
    return Omega;
}




cv::vector<cv::KeyPoint> getKeyPoints(cv::Mat &img, cv::Ptr<cv::FeatureDetector> detector){
    cv::vector<cv::KeyPoint> K;
    detector->detect(img, K);
    return K;
}




- (NSUInteger)application:(UIApplication *)application supportedInterfaceOrientationsForWindow:(UIWindow *)window
{
    return UIInterfaceOrientationMaskPortrait;
}

// Member functions for converting from cvMat to UIImage
- (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}

@end
