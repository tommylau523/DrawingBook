//
//  ViewController.h
//  DrawingBook
//
//  Created by Tommy Lau on 11/17/15.
//  Copyright © 2015 Tommy Lau. All rights reserved.
//

#import <UIKit/UIKit.h>


#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#import "opencv2/highgui/ios.h"
#import <UIKit/UIKit.h>
#import <CoreMotion/CoreMotion.h>
#endif

@interface ViewController : UIViewController<CvVideoCameraDelegate>
- (IBAction)resetMaxValues:(id)sender;

@property (strong, nonatomic) CMMotionManager *motionManager;
@property(readonly, nonatomic) CMAcceleration* userAcceleration;



@end

