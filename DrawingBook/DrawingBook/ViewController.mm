//
//  ViewController.m
//  DrawingBook
//
//  Created by Tommy Lau on 11/17/15.
//  Copyright © 2015 Tommy Lau. All rights reserved.
//

#import "ViewController.h"

#ifdef __cplusplus
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/ios.h"
#include <opencv2/features2d/features2d.hpp>
#endif

// Include stdlib.h and std namespace so we can mix C++ code in here
#include <stdlib.h>
#include <iostream>
using namespace std;


@interface ViewController (){
    UIImageView *liveView_; // Live output from the camera
    CvVideoCamera *videoCamera_; // OpenCV wrapper class to simplfy camera access through AVFoundation;
    cv::vector<cv::KeyPoint> K_template;
    cv::Mat template_im;
    cv::Mat template_descriptor;
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> extractor;
    cv::Ptr<cv::BFMatcher> matcher;
}
@property (nonatomic, retain) CvVideoCamera* videoCamera;
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    UIImage *template_image = [UIImage imageNamed:@"template.jpg"];
    detector = new cv::OrbFeatureDetector(600);
    extractor = new cv::OrbDescriptorExtractor;
    matcher = new cv::BFMatcher(cv::NORM_L2, true);
    template_im = [self cvMatFromUIImage:template_image];
    K_template = getKeyPoints(template_im, detector);
    extractor->compute(template_im, K_template, template_descriptor);
    
    
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
    cv::Mat descriptor_output;
    std::vector<cv::DMatch> matches;
    
    cvtColor(image, image_copy, CV_BGRA2BGR);
    cvtColor(image_copy, gray, CV_BGRA2GRAY);
    cvtColor(template_im, template_gray, CV_BGRA2GRAY);
    
    
    cv::vector<cv::KeyPoint> K = getKeyPoints(image_copy, detector);
    extractor->compute(image_copy, K, descriptor_output);
    
    matcher->match(descriptor_output, template_descriptor, matches);
    
    cv::drawMatches(gray, K, template_gray, K_template, matches, image);
    //cvtColor(image_copy, image, CV_BGR2BGRA);
}




cv::vector<cv::KeyPoint> getKeyPoints(cv::Mat &img, cv::Ptr<cv::FeatureDetector> detector){
    cv::vector<cv::KeyPoint> K;
    detector->detect(img, K);
    return K;
}




- (NSUInteger)application:(UIApplication *)application supportedInterfaceOrientationsForWindow:(UIWindow *)window
{
    return UIInterfaceOrientationMaskLandscape;
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