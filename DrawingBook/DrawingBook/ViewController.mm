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
#import <UIKit/UIKit.h>
#import <CoreMotion/CoreMotion.h>
//#include "armadillo"
#endif

// Include stdlib.h and std namespace so we can mix C++ code in here

using namespace std;

@interface ViewController (){
    UIImageView *liveView_; // Live output from the camera
    CvVideoCamera *videoCamera_; // OpenCV wrapper class to simplfy camera access through AVFoundation;
    cv::vector<cv::KeyPoint> template_keypoints;
    cv::Mat template_im, template_gray, template_copy;
    cv::Mat template_descriptor;
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> extractor;
    cv::Ptr<cv::BFMatcher> matcher;
    cv::Mat intrinsics;
    cv::Mat distCoeffs;
    cv::Mat lastImage;

    std::vector<cv::Point2f> prev_scene_corners;
    double prev_scene_updated;
    CMAcceleration acc;
    std::vector<cv::KalmanFilter> KF;
    std::vector<cv::Point2f> obj_corners;
    
    bool started, first;
}
@property (nonatomic, retain) CvVideoCamera* videoCamera;
@end





@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    UIImage *template_image = [UIImage imageNamed:@"template.jpg"];
    
    detector = new cv::OrbFeatureDetector(1000);
    extractor = new cv::OrbDescriptorExtractor;
    matcher = new cv::BFMatcher(cv::NORM_HAMMING, true);
    template_im = [self cvMatFromUIImage:template_image];
    cvtColor(template_im, template_gray, CV_BGRA2GRAY);
    cvtColor(template_im, template_copy, CV_BGRA2BGR);
    template_keypoints = getKeyPoints(template_im, detector);
    extractor->compute(template_im, template_keypoints, template_descriptor);
    intrinsics = cv::Mat::zeros(3,3,CV_64F);
    intrinsics.at<double>(0,0) = 2871.8995;
    intrinsics.at<double>(1,1) = 2871.8995;
    intrinsics.at<double>(2,2) = 1;
    intrinsics.at<double>(0,2) = 1631.5;
    intrinsics.at<double>(1,2) = 1223.5;
    distCoeffs = cv::Mat(5,1,cv::DataType<double>::type);
    distCoeffs.at<double>(0) = -.0008211;
    distCoeffs.at<double>(1) = 0.640757;
    distCoeffs.at<double>(2) = 0;
    distCoeffs.at<double>(3) = 0;
    distCoeffs.at<double>(4) = -1.7248;
    first = true;

    started = false;
    
    self.motionManager = [[CMMotionManager alloc] init];
    self.motionManager.accelerometerUpdateInterval = .2;
    self.motionManager.gyroUpdateInterval = .2;
    
    [self.motionManager startAccelerometerUpdatesToQueue:[NSOperationQueue currentQueue]
                                             withHandler:^(CMAccelerometerData  *accelerometerData, NSError *error) {
                                                 [self outputAccelertionData:accelerometerData.acceleration];
                                                 if(error){
                                                     
                                                     NSLog(@"%@", error);
                                                 }
                                             }];
    
    
    obj_corners = std::vector<cv::Point2f> (4);
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( template_im.cols, 0 );
    obj_corners[2] = cvPoint( template_im.cols, template_im.rows );
    obj_corners[3] = cvPoint( 0, template_im.rows );
    
    
    // 1. Setup the your OpenCV view, so it takes up the entire App screen......
    int view_width = self.view.frame.size.width;
    int view_height = (960*view_width)/540; // Work out the viw-height assuming 640x480 input
    int view_offset = (self.view.frame.size.height - view_height)/2;
    liveView_ = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, view_offset, view_width, view_height)];
    [self.view addSubview:liveView_];
    
    
    // 4. Initialize the camera parameters and start the camera (inside the App)
    videoCamera_ = [[CvVideoCamera alloc] initWithParentView:liveView_];
    videoCamera_.delegate = self;
    videoCamera_.defaultFPS = 24;
    videoCamera_.grayscaleMode = NO;
    videoCamera_.rotateVideo = YES;
    
    // This chooses whether we use the front or rear facing camera
    videoCamera_.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
    
    // This is used to set the image resolution
    videoCamera_.defaultAVCaptureSessionPreset = AVCaptureSessionPresetiFrame960x540;
    
    // This is used to determine the device orientation
    videoCamera_.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationLandscapeLeft;
    
    // This starts the camera capture
    [videoCamera_ start];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (void)resetMaxValues:(id)sender{
}

std::vector<cv::KalmanFilter> start_KF(std::vector<cv::Point2f> pts){
    std::vector<cv::KalmanFilter> KF(4);
    for(int i = 0; i < KF.size(); i++){
        KF[i] = cv::KalmanFilter(4,2,0);
        cout << KF[i].statePre.size() << endl;
        KF[i].statePre.at<float>(0) = pts[i].x;
        KF[i].statePre.at<float>(1) = pts[i].y;
        KF[i].statePre.at<float>(2) = 0;
        KF[i].statePre.at<float>(3) = 0;
        
        KF[i].transitionMatrix = *(cv::Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
        cv::setIdentity(KF[i].measurementMatrix);
        cv::setIdentity(KF[i].processNoiseCov, cv::Scalar::all(.1));
        cv::setIdentity(KF[i].measurementNoiseCov, cv::Scalar::all(50));
        cv::setIdentity(KF[i].errorCovPost, cv::Scalar::all(.1));
    }
    return KF;
}

std::vector<cv::Point2f>  KF_estimate(std::vector<cv::KalmanFilter> KF, std::vector<cv::Point2f> pts){
    std::vector<cv::Point2f> corrpt(KF.size());
    for(int i = 0; i < KF.size(); i++){
        cv::Mat_<float> m(2,1);
        m(0) = pts[i].x;
        m(1) = pts[i].y;
        cv::Mat est = KF[i].correct(m);
        corrpt[i] = cv::Point2f(est.at<float>(0), est.at<float>(1));
    }
    return corrpt;
}

std::vector<cv::Point2f>  KF_predict(std::vector<cv::KalmanFilter> KF){
    std::vector<cv::Point2f> predpts(KF.size());
    for(int i = 0; i < KF.size(); i++){
        cv::Mat pred = KF[i].predict();
        predpts[i] = cv::Point2f(pred.at<float>(0), pred.at<float>(1));
    }
    return predpts;
}


-(void)outputAccelertionData:(CMAcceleration)acceleration
{
    acc = acceleration;
}

- (void)processImage:(cv::Mat&) image;
{
    // Do some OpenCV stuff with the image
    cv::Mat image_copy;
    cv::Mat image_descriptor;
    std::vector<cv::DMatch> matches;
    
    cvtColor(image, image_copy, CV_BGRA2BGR);
    
    
    @try{
        
        
        
        
        //Image Matching
        cv::vector<cv::KeyPoint> image_keypoints = getKeyPoints(image_copy, detector);
        extractor->compute(image_copy, image_keypoints, image_descriptor);
        if(image_descriptor.cols < 20){
            return; //terminate if less than 20 keypoints
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
        {
            if( matches[i].distance < 4*min_dist )
            {
                good_matches.push_back( matches[i]);
            }
        }
        cv::vector<cv::Point3f> source;
        cv::vector<cv::Point2f> source2;
        cv::vector<cv::Point2f> dest;
        for(int i = 0; i < good_matches.size(); i++){
            source.push_back(cv::Point3f(template_keypoints[good_matches[i].queryIdx].pt.x,
                                         template_keypoints[good_matches[i].queryIdx].pt.y,
                                         0));
            source2.push_back(template_keypoints[good_matches[i].queryIdx].pt);
            dest.push_back(image_keypoints[good_matches[i].trainIdx].pt);
        }
        
        if(good_matches.size() < 50){
            return; //terminate if not enough good matches
        }
        
        cv::Mat inliers_mask;
        cv::Mat H = cv::findHomography(source2, dest, CV_RANSAC, 5, inliers_mask);
        
        std::vector<cv::Point2f> scene_corners(4);
        
        cv::perspectiveTransform( obj_corners, scene_corners, H);
        
        int numInliers = 0;
        float distance = 0;
        for(int i = 0; i < inliers_mask.rows; i++){
            if((int)inliers_mask.at<uchar>(i) == 1){
                numInliers += 1;
                distance += good_matches[i].distance;
            }
        }
        
        distance /= numInliers;
        
        
        double accelerationSq = acc.x*acc.x + acc.y*acc.y + acc.z*acc.z;
        double time = CACurrentMediaTime();
        if(numInliers < good_matches.size() * .15 || distance > 55 || accelerationSq > 1.1){
            if (first){
                return;
            }
            else {
                if( time - prev_scene_updated < 1){
                    scene_corners = prev_scene_corners;
                }
                else{
                    return;
                }
            }
        }
        
        cv::Mat camera_template;
        cv::warpPerspective(image_copy, camera_template, H.inv(), template_copy.size());
        
//        if(!started){
//            started = true;
//            KF = start_KF(scene_corners);
//            cout << KF[0].statePre.size() << endl;
//            prev_scene_corners = scene_corners;
//        }
        
        //-- Draw lines between the corners
        cv::line( image_copy, scene_corners[0], scene_corners[1], cv::Scalar(0, 255, 0), 4 );
        cv::line( image_copy, scene_corners[1], scene_corners[2], cv::Scalar( 0, 255, 0), 4 );
        cv::line( image_copy, scene_corners[2], scene_corners[3], cv::Scalar( 0, 255, 0), 4 );
        cv::line( image_copy, scene_corners[3], scene_corners[0], cv::Scalar( 0, 255, 0), 4 );
        
        
//        double a = dist(scene_corners[0],scene_corners[1]);
//        double b = dist(scene_corners[1],scene_corners[2]);
//        double c = dist(scene_corners[2],scene_corners[3]);
//        double d = dist(scene_corners[3],scene_corners[0]);
//        
//        if(  max(a,max(b,max(c,d))) > 1100 ||  min(a,min(b,min(c,d))) < 100){
//            return;
//        }
//        
//        cv::vector<cv::Point2f> predPts, estPts;
//        predPts = KF_predict(KF);
//        estPts = KF_estimate(KF, scene_corners);
//        
//        
//        float scene_error = 0;
//        for(int i = 0; !first && i < scene_corners.size(); i++){
//            scene_error += dist(scene_corners[i], prev_scene_corners[i]);
//        }
//        if(scene_error > 120){
//            return;
//        }
//        else{
//            prev_scene_corners = scene_corners;
//            prev_scene_updated = time;
//        }
//        
        
        cv::Mat rvec, tvec;
        std::vector<cv::Point3f> proj_corners(4);
        std::vector<cv::Point2f> scene_proj_corners(4);
        
        float x = 120;
        float y = 385;
        float w = 188;
        float h = -200;
        proj_corners[0] = cv::Point3f(0,0,0);
        proj_corners[1] = cv::Point3f( template_im.cols, 0, 0 );
        proj_corners[2] = cv::Point3f( template_im.cols, template_im.rows, 0 );
        proj_corners[3] = cv::Point3f( 0, template_im.rows, 0 );
        
        //either solve with scene_corners or estPts
        cv::solvePnP(proj_corners, scene_corners, intrinsics, distCoeffs, rvec, tvec);
        
        std::vector<cv::Point3f> cube_corners(8);
        //left face
        cube_corners[0] = cv::Point3f(x + w, y, h);
        cube_corners[1] = cv::Point3f(x + w, y, h-w );
        cube_corners[2] = cv::Point3f(x + w, y + w, h-w );
        cube_corners[3] = cv::Point3f(x + w, y + w, h );
        cube_corners[4] = cv::Point3f(x + w + w, y, h);
        cube_corners[5] = cv::Point3f(x + w + w, y, h-w );
        cube_corners[6] = cv::Point3f(x + w + w, y + w, h-w );
        cube_corners[7] = cv::Point3f(x + w + w, y + w, h );
        
        std::vector<cv::Point2f> cube_proj_corners;
        
        cv::projectPoints(cube_corners, rvec, tvec, intrinsics, distCoeffs, cube_proj_corners);
        cv::projectPoints(proj_corners, rvec, tvec, intrinsics, distCoeffs, scene_proj_corners);
        
        cv::line( image_copy, scene_proj_corners[0], scene_proj_corners[1], cv::Scalar(255, 0, 255), 1 );
        cv::line( image_copy, scene_proj_corners[1], scene_proj_corners[2], cv::Scalar(255, 0, 255), 1 );
        cv::line( image_copy, scene_proj_corners[2], scene_proj_corners[3], cv::Scalar(255, 0, 255), 1 );
        cv::line( image_copy, scene_proj_corners[3], scene_proj_corners[0], cv::Scalar(255, 0, 255), 1 );
        
        
        //SQUARE 1
        cv::Mat cameraSq1;
        cv::Rect sq1(x, y, w, w);
        cameraSq1 = cv::Mat(camera_template, sq1).clone();
        std::vector<cv::Point2f> sq1_proj(4);
        std::vector<cv::Point2f> sq1_pts(4);
        sq1_proj[0] = cube_proj_corners[1];
        sq1_pts[0] = cv::Point2f(0 ,0);
        sq1_proj[1] = cube_proj_corners[0];
        sq1_pts[1] = cv::Point2f(cameraSq1.cols ,0);
        sq1_proj[2] = cube_proj_corners[2];
        sq1_pts[2] = cv::Point2f(0 , cameraSq1.rows);
        sq1_proj[3] = cube_proj_corners[3];
        sq1_pts[3] = cv::Point2f(cameraSq1.cols, cameraSq1.rows);
        
        //SQUARE 2
        cv::Mat cameraSq2;
        cv::Rect sq2(x+w, y-w, w, w);
        cameraSq2 = cv::Mat(camera_template, sq2).clone();
        std::vector<cv::Point2f> sq2_proj(4);
        std::vector<cv::Point2f> sq2_pts(4);
        sq2_proj[0] = cube_proj_corners[1];
        sq2_pts[0] = cv::Point2f(0 ,0);
        sq2_proj[1] = cube_proj_corners[5];
        sq2_pts[1] = cv::Point2f(cameraSq2.cols ,0);
        sq2_proj[2] = cube_proj_corners[0];
        sq2_pts[2] = cv::Point2f(0 , cameraSq2.rows);
        sq2_proj[3] = cube_proj_corners[4];
        sq2_pts[3] = cv::Point2f(cameraSq2.cols, cameraSq2.rows);
        
        
        //SQUARE 3
        cv::Mat cameraSq3;
        cv::Rect sq3(x+w, y, w, w);
        cameraSq3 = cv::Mat(camera_template, sq3).clone();
        std::vector<cv::Point2f> sq3_proj(4);
        std::vector<cv::Point2f> sq3_pts(4);
        sq3_proj[0] = cube_proj_corners[0];
        sq3_pts[0] = cv::Point2f(0 ,0);
        sq3_proj[1] = cube_proj_corners[4];
        sq3_pts[1] = cv::Point2f(cameraSq3.cols ,0);
        sq3_proj[2] = cube_proj_corners[3];
        sq3_pts[2] = cv::Point2f(0 , cameraSq3.rows);
        sq3_proj[3] = cube_proj_corners[7];
        sq3_pts[3] = cv::Point2f(cameraSq3.cols, cameraSq3.rows);
        
        //SQUARE 4
        cv::Mat cameraSq4;
        cv::Rect sq4(x+w, y+w, w, w);
        cameraSq4 = cv::Mat(camera_template, sq4).clone();
        std::vector<cv::Point2f> sq4_proj(4);
        std::vector<cv::Point2f> sq4_pts(4);
        sq4_proj[0] = cube_proj_corners[3];
        sq4_pts[0] = cv::Point2f(0 ,0);
        sq4_proj[1] = cube_proj_corners[7];
        sq4_pts[1] = cv::Point2f(cameraSq4.cols ,0);
        sq4_proj[2] = cube_proj_corners[2];
        sq4_pts[2] = cv::Point2f(0 , cameraSq4.rows);
        sq4_proj[3] = cube_proj_corners[6];
        sq4_pts[3] = cv::Point2f(cameraSq4.cols, cameraSq4.rows);
        
        //SQUARE 5
        cv::Mat cameraSq5;
        cv::Rect sq5(x+w, y+w+w, w, w);
        cameraSq5 = cv::Mat(camera_template, sq5).clone();
        std::vector<cv::Point2f> sq5_proj(4);
        std::vector<cv::Point2f> sq5_pts(4);
        sq5_proj[0] = cube_proj_corners[2];
        sq5_pts[0] = cv::Point2f(0 ,0);
        sq5_proj[1] = cube_proj_corners[6];
        sq5_pts[1] = cv::Point2f(cameraSq5.cols ,0);
        sq5_proj[2] = cube_proj_corners[1];
        sq5_pts[2] = cv::Point2f(0 , cameraSq5.rows);
        sq5_proj[3] = cube_proj_corners[5];
        sq5_pts[3] = cv::Point2f(cameraSq5.cols, cameraSq5.rows);
        
        //SQUARE 6
        cv::Mat cameraSq6;
        cv::Rect sq6(x+w+w, y, w, w);
        cameraSq6 = cv::Mat(camera_template, sq6).clone();
        std::vector<cv::Point2f> sq6_proj(4);
        std::vector<cv::Point2f> sq6_pts(4);
        sq6_proj[0] = cube_proj_corners[4];
        sq6_pts[0] = cv::Point2f(0 ,0);
        sq6_proj[1] = cube_proj_corners[5];
        sq6_pts[1] = cv::Point2f(cameraSq6.cols ,0);
        sq6_proj[2] = cube_proj_corners[7];
        sq6_pts[2] = cv::Point2f(0 , cameraSq6.rows);
        sq6_proj[3] = cube_proj_corners[6];
        sq6_pts[3] = cv::Point2f(cameraSq6.cols, cameraSq6.rows);
        
        //draw_background(H, image_copy, template_im.cols, template_im.rows);
        overlay_image(image_copy, cameraSq3, sq3_pts, sq3_proj);
        overlay_image(image_copy, cameraSq1, sq1_pts, sq1_proj);
        overlay_image(image_copy, cameraSq2, sq2_pts, sq2_proj);
        overlay_image(image_copy, cameraSq4, sq4_pts, sq4_proj);
        overlay_image(image_copy, cameraSq5, sq5_pts, sq5_proj);
        overlay_image(image_copy, cameraSq6, sq6_pts, sq6_proj);
        if(first){
            first = false;
        }
    } @catch(NSException *theException) {
        cout << theException.name << endl;
        cout << theException.reason << endl;
    }
    
    cvtColor(image, lastImage, CV_BGRA2BGR);
    cvtColor(image_copy, image, CV_BGR2BGRA);


}

double dist(cv::Point2f a, cv::Point2f(b)){
    return cv::norm(a-b);
}

void overlay_image(cv::Mat image, cv::Mat square, std::vector<cv::Point2f> sq_pts, std::vector<cv::Point2f> sq_proj){
    cv::Point2f x = sq_proj[1] - sq_proj[0];
    cv::Point2f y = sq_proj[2] - sq_proj[0];
    if(x.x*y.y - x.y*y.x < 0){
        cv::Mat H = cv::findHomography(sq_pts, sq_proj);
        cv::warpPerspective(square, square, H, image.size());
        cv::Mat square_gray, mask, mask_inv, roi, mask_fg, mask_bg;
        image.copyTo(roi);
        
        //Now create a mask of logo and create its inverse mask also
        cv::cvtColor(square, square_gray, CV_BGR2GRAY);
        cv::threshold(square_gray, mask, 10, 255, cv::THRESH_BINARY);
        cv::bitwise_not(mask, mask_inv);
        
        //Now black-out the area of logo in ROI
        cv::bitwise_and(roi,roi, mask_bg, mask_inv);
        
        //Take only region of logo from logo image.
        cv::bitwise_and(square, square, mask_fg, mask);
        
        //Put logo in ROI and modify the main image
        cv::add(mask_bg, mask_fg, image);
    }
}

void draw_background(cv::Mat H, cv::Mat image, int w, int h){
    
    cv::Mat bg(h, w, CV_8UC3, cv::Scalar(255,255,255));
    
    
    cv::warpPerspective(bg, bg, H, image.size());
    cv::Mat square_gray, mask, mask_inv, roi, mask_fg, mask_bg;
    image.copyTo(roi);
    
    //Now create a mask of logo and create its inverse mask also
    cv::cvtColor(bg, square_gray, CV_BGR2GRAY);
    cv::threshold(square_gray, mask, 10, 255, cv::THRESH_BINARY);
    cv::bitwise_not(mask, mask_inv);
    
    //Now black-out the area of logo in ROI
    cv::bitwise_and(roi,roi, mask_bg, mask_inv);
    
    //Take only region of logo from logo image.
    cv::bitwise_and(bg, bg, mask_fg, mask);
    
    //Put logo in ROI and modify the main image
    cv::add(mask_bg, mask_fg, image);
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
