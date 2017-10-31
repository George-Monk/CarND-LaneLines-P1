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

the next step was to mask the image to remove the unwanted information, outside the area of interest (the road ahead). This was achieved by creating a trapezium-shaped polygon which scaled with the dimensions of the image/frame, to select the lane ahead only.

The Hough Transform was then performed on the image to identify lines within the masked image area. 

The next step was perhaps the most involved and required significant modification of the draw_lines function. The multiple detected lines from the Hough Transform are taken by the draw_lines function where their gradients are first analysed to determine if the line is part of the detected right line, or left line. As the x and y values are counted from the top left in image processing, this meant that a line with a positive gradient represented a line along a road marking on the right side of the vehicle and a negative gradient represented a line marking along the left.

After separating the lines based on their gradients, their line start/end coordinates were stored in an array where they could be used to determine the overall average description of each line so that a single line could be drawn for each. In this instance, two methods were implemented:

  - Calculation of line description using 'y = mx + c' 

  - Calculation of the start and end of a line, using 'y = mx + c' for the start of the line, assuming the start of the line to be at       the bottom of the image (y = image height), and the end of the line being the coordinates with the largest/smallest value of 'x'         within the stored array for each frame (largest for left line and smallest for right line).




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
