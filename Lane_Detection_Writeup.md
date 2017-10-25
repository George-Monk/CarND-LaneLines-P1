# **Finding Lane Lines on the Road** 

## Lane Detection Writeup
---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[image2]: ./test_images/ProcessedImages/ProcessedImage1.jpg "Solid White Curve"
[image3]: ./test_images/ProcessedImages/ProcessedImage2.jpg "Solid White Right"
[image4]: ./test_images/ProcessedImages/ProcessedImage3.jpg "Solid Yellow Curve"
[image5]: ./test_images/ProcessedImages/ProcessedImage4.jpg "Solid Yellow Curve 2"
[image6]: ./test_images/ProcessedImages/ProcessedImage5.jpg "Solid Yellow Left"
[image7]: ./test_images/ProcessedImages/ProcessedImage6.jpg "White Car Lane Switch"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps. Firstly, the images were converted to grayscale (below), with each pixel being represented by a value between 0-255. 

![alt text][image1]

--------------------------------------------------------------------------------------------------------------
Next, Gaussian smoothing was applied, using a kernel size of 5 in this case. The smoothing helps reduce noise within the image to aid with the line detection process.

Following this, Canny Edge detection was performed, using a lower bound of 50 and an upper of 150, this highlights the areas of an image where there is a large delta in pixel value along consecutive, neighboring pixels which constitutes the edge of an object within an image (including the outline of the lane lines).

the next step was to mask the image to remove the unwanted information, outside the area of interest (the road ahead). This was achieved by creating a trapezium-shaped polygon to select the lane ahead only.



In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 



![alt text][image2]

--------------------------------------------------------------------------------------------------------------

![alt text][image3]

--------------------------------------------------------------------------------------------------------------

![alt text][image4]

--------------------------------------------------------------------------------------------------------------

![alt text][image5]

--------------------------------------------------------------------------------------------------------------

![alt text][image6]

--------------------------------------------------------------------------------------------------------------

![alt text][image7]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 
No predictive plot, so loss of line detection could lead to blank/flicker (no plot)

Method does not perform plausability check based on average of last few frames (complex shadows/road markings may lead to significant error)

Extensive use of lists (could be optimised)

Manual calculation of gradient and 'c' (y=mx+c), could use library function such as Numpy Polyfit/Poly1d/Linspace method.

Average is over 15 frames (could result in slow response for fast corners)

Although the defined mask scales with video image size, it has been calibrated and fixed to the camera position in the given examples. It could be adjusted to look at a more confined area ( it occasionaly detects distant vehicles as road markings, resulting in line average creep). Additionaly, it could be programmed to automatically adjust for corners by bending left/right when a turn is detected.




Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
