# DrawingBook
16-423 Project by Tommy Lau and Will Willsey.

#Summary
In this  project we plan on developing an application on iOS devices in order to project augmented reality 3D objects onto a coloring book, where the colors filled in are projected onto the 3D object itself. We plan on mapping surfaces and textures of the 3D models to patterns on a coloring book or coloring sheet, and any coloring onto the pattern will then be reflected and projected to the augmented reality 3D model. Our goal is to familiar ourselves with copmuter vision algorithms and iOS development in the process while developing a creative upgrade to ordinary coloring games.

#Background
The coloring sheet will have a pattern that the pinhole camera should be able to recognize and therefore used to compute neccesary parameters including homography transforms and camera extrinsic matrices. We will need feature points detectors, descriptor generators and matchers to find matchings to the coloring sheet, and compute the neccessary parameters using our matchings. Potential feature point detectors we can use are SIFT, FAST or BRIEF. The Aramadillo Library will be used for matrices computations. We might also need to track the object movement using the Lucas-Kanade algorithm, and project textures onto the 3D model using other libraries. Learning algorithms can also be implemented, depending on the performance.

#The Challenge
The problem revolves around many of the problems and algorithms we learn about computer vision. This includes object detection, object tracking, computing parameters for warps and the camera, and even computer graphics projections. We will need to integrate many aspects of computer vision in order to develop a functional and robust application. This project introduces computer vision to an existing tool, in our case a coloring book, and improve it by applying special effects that can only be done using computer visioni algorithms on mobile devices.

#Goals and Deliverables
We plan on creating a robust algorithm that allows us to detect a coloring sheet, project a 3D model of a certain object onto the sheet, and then fill in colorings depending on the colorings on the sheet. We plan on achieving this for simple objects such as spheres, cubes, and hope to march on to implement the process to more complicated objects such as houses, buildings, or even cartoon characters. We want to be able to apply computer vision concepts and algorithms to develop a robust program. Our final goal is to be able to notice the changes in the 3D model, when the camera is facing the coloring sheet while it is getting filled in. We should notice colors getting applied onto the 3D model. We believe that our project is within the correct scope given our timeframe, and that we can  acheive a coloring book of objects of vaarious complexity.

#Schedule
We have approximately 4 weeks to complete the project, with the checkpoint due about 2 weeks from now. Below are some goals we would like to achieve within the time schedule.

Week 1: Recognize the coloring sheet and implement algorithms that can compute the neccesary warps and parameters.

Week 2: Recognize patterns on the coloring sheet and project the 3D model onto the sheeet. (Check Point)

Week 3: Perform tracking of any movement of the paper, and identify any coloring or pen strokes on the paper. Project colors onto the model.

Week 4: Finish color projections and fine tune neccesaray algorithms for robustness. 
