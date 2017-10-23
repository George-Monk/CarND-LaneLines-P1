
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Project: **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, consult the forums for more troubleshooting tips.**  

# ## Import Packages

# In[1]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import glob
import imageio
from collections import deque, Counter
# Needed for Jupyter?
get_ipython().run_line_magic('matplotlib', 'inline')

# Set directory to Project Base Dir due to the unique way in which VS functions
ProjectDir = "C:/GitRepo/Udacity_Training/Finding-Lane-Lines-on-the-Road/CarND-LaneLines-P1"
os.chdir(ProjectDir)
cwd = os.getcwd()
print('current working dir is: '+ cwd)

#Create array for storing gradients
leftLineGradArray = []
rightLineGradArray = []

# Create array for storing average x values
xLeftAverageArray = []
xRightAverageArray = []

xMovingAverage = deque([], maxlen = 15)

# Counter for tracking frame number
count = Counter()

# ## Read in an Image

# In[2]:

#reading in an image
image = mpimg.imread('test_images/solidWhiteCurve.jpg')

# Initialise File Count (for File Export)
fileCount = 1

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# ## Ideas for Lane Detection Pipeline

# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images
# `cv2.cvtColor()` to grayscale or change color
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# ## Helper Functions

# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[3]:


import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines,leftLineGradArray, rightLineGradArray, xLeftAverageArray, xRightAverageArray, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    imgWidth = img.shape[1]
    imgHeight = img.shape[0]

    x_left  = []
    y_left  = []

    x_right = []
    y_right = []


    # Update Frame Count
    count.update('F')
    frameCount=count.get('F')
    print('')
    print ('Frame Count : %s'%frameCount)


    # Append to left line array
    leftGradAppend = leftLineGradArray.append

    # Append to right line array
    rightGradAppend = rightLineGradArray.append

    xLeftAverageAppend = xLeftAverageArray.append
    xRightAverageAppend = xRightAverageArray.append
    

    for line in lines:
        for x1,y1,x2,y2 in line:

            # Calculate gradient
            m = (y2-y1)/(x2-x1)
            
            
            # If gradient is positive, line is right line
            if 5 > m > 0.5:
                #print('right Line gradient: %s' %(m))
                rightLineGrad = m
                rightGradAppend(m)  
                
                # Add right x & y coordinates to Array
                x_right += [x1, x2]
                y_right += [y1, y2]

            # If gradient is negative, line is left line
            elif -5 < m < -0.5:
                #print('left line gradient: %s' % (m))
                leftLineGrad = m
                leftGradAppend(m)

                # Add left x & y coordinates to Array
                x_left += [x1, x2]
                y_left += [y1, y2]
            
            
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


    # If no line is detected, do not process & print error
    if not x_left :
        print('Empty x_left list, No lines found in this image')
    elif not x_right :
        print('Empty x_right list, No lines found in this image')
    elif not y_left :
        print('Empty y_left list, No lines found in this image')
    elif not y_right :
        print('Empty y_right list, No lines found in this image')

    # If line is detected, continue
    else :
        # Calculate average x & y for right line from the array of all detected lines in frame
        xRightAverage = sum(x_right)/len(x_right)
        yRightAverage = sum(y_right)/len(y_right)
        print('Right Line average x is : %s'%xRightAverage)
        print('Right Line average y is : %s'%yRightAverage)

        # Calculate average x & y for left line from the array of all detected lines in frame
        xLeftAverage = sum(x_left)/len(x_left)
        yLeftAverage = sum(y_left)/len(y_left)

        # Create array to store the average over multiple frames, using the value from single frames above
        xLeftAverageAppend (int(round(xLeftAverage)))
        xRightAverageAppend (int(round(xRightAverage)))

        # calculate average gradient of right line
        rightGradAverage = sum(rightLineGradArray)/len(rightLineGradArray)
        print('Manual Right line Gradient Average: %s'%rightGradAverage)
        
        #calculate average gradient of left line
        leftGradAverage = sum(leftLineGradArray)/len(leftLineGradArray)
        print('Manual left Line Gradient Average: %s'%leftGradAverage)
    
        # Right Line: y=mx+c, therefore, to calculate c, we need c=y-mx
        cRightAverage = yRightAverage - (rightGradAverage*xRightAverage)
        print('c right average :   %s'%cRightAverage)

        # Left Line:  y=mx+c, therefore, to calculate c, we need c=y-mx
        cLeftAverage = yLeftAverage - (leftGradAverage*xLeftAverage)
        print('c left average :   %s'%cLeftAverage)
        print('yLeftAverage : %s'%yLeftAverage)
        print('xLeftAverage : %s'%xLeftAverage)
        print('')

        # Rearanging for x: x = (y-c)/m
        #Start of line should be at y = image height
        x_RightLineStart = int(round((imgHeight-cRightAverage)/rightGradAverage))
        x_LeftLineStart = int(round((imgHeight-cLeftAverage)/leftGradAverage))


        #End of line should be at y = 320
        y_LineEnd = 320

        # Find end of each line by largest/smallest value in line array
        x_RightLineEnd = min(x_right)
        x_LeftLineEnd = max(x_left)


        # Calculate end of line by extrapolating using y=mx+c
        x_ExtrapRightLineEnd = int(round((y_LineEnd-cRightAverage)/rightGradAverage))
        x_ExtrapLeftLineEnd = int(round((y_LineEnd-cLeftAverage)/leftGradAverage))

        #Take conglomerated average of extrapolated expected value and min/max of array (end of detected line)
        # A weighting has been added to the more stable y=mx+c calculated x values
        x_RightLineEndConglomerate = int(round((x_RightLineEnd + x_ExtrapRightLineEnd + x_ExtrapRightLineEnd)/3))
        x_LeftLineEndConglomerate = int(round((x_LeftLineEnd + x_ExtrapLeftLineEnd + x_ExtrapLeftLineEnd)/3))

        # Append values to moving average array
        xMovingAverage.append((x_RightLineStart, x_RightLineEndConglomerate, x_LeftLineStart, x_LeftLineEndConglomerate))
        
        # Calculate sum of terms within array
        x_LineAverageSum = np.sum(xMovingAverage, -2)
        print('x_LineAverageSum : %s'%x_LineAverageSum)

        # If frame count is below the averaging window threshold (sliding window size), display raw values
        if frameCount <= 15 :
            cv2.line(img, (x_RightLineStart, imgHeight), (x_ExtrapRightLineEnd , y_LineEnd), color, thickness)
            cv2.line(img, (x_LeftLineStart, imgHeight), (x_ExtrapLeftLineEnd , y_LineEnd), color, thickness)

        # Otherwise, use averaged values
        else :
            x_RightLineStartAverage = int(x_LineAverageSum[0]/len(xMovingAverage))
            x_RightLineEndAverage = int(x_LineAverageSum[1]/len(xMovingAverage))
            x_LeftLineStartAverage = int(x_LineAverageSum[2]/len(xMovingAverage))
            x_LeftLineEndAverage = int(x_LineAverageSum[3]/len(xMovingAverage))

            cv2.line(img, (x_RightLineStartAverage, imgHeight), (x_RightLineEndAverage , y_LineEnd), color, thickness)
            cv2.line(img, (x_LeftLineStartAverage, imgHeight), (x_LeftLineEndAverage , y_LineEnd), color, thickness)

    return rightLineGradArray, leftLineGradArray, xLeftAverageArray, xRightAverageArray

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    draw_lines(line_img, lines, leftLineGradArray, rightLineGradArray, xLeftAverageArray, xRightAverageArray)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def detect_lanes(image, fileCount):

    # Convert Image to Greyscale
    grey = grayscale(image)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    BlurredImage = gaussian_blur(grey, kernel_size)

    # Define parameters for Canny Edge Detection
    low_threshold = 50
    high_threshold = 150
    BlurredImageCanny = canny(BlurredImage, low_threshold, high_threshold)

    # Mask Edges of Defined Polygon
    imshape = image.shape
    vertices = np.array([[(150,imshape[0]),(450, 320), (imshape[1]-450, 320), (imshape[1]-50,imshape[0])]], dtype=np.int32)
    MaskedImage = region_of_interest(BlurredImageCanny, vertices)

    # Hough Transform Parameters
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = 0.027*np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 5     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10 #minimum number of pixels making up a line
    max_line_gap = 2    # maximum gap in pixels between connectable line segments
    DetectedLines = hough_lines(MaskedImage, rho, theta, threshold, min_line_length, max_line_gap)

    # Merge Images
    ProcessedImage = weighted_img(DetectedLines, image, α=0.8, β=1., λ=0.)

    outputDir = 'test_images/ProcessedImages/'

    figure = plt.figure(figsize=(20,10))

    plotImage=figure.add_subplot(1,4,1)
    plt.imshow(image)
    plotImage.set_title('Unprocessed Image')

    plotImage=figure.add_subplot(1,4,2)
    plt.imshow(BlurredImageCanny)
    plotImage.set_title('Post-Canny Image')

    plotImage=figure.add_subplot(1,4,3)
    plt.imshow(MaskedImage)
    plotImage.set_title('Masked Image')

    plotImage=figure.add_subplot(1,4,4)
    plt.imshow(ProcessedImage)
    plotImage.set_title('Processed Image')

    plt.savefig(outputDir + 'ProcessedImage' + str(fileCount) + '.jpg')
    
    return BlurredImageCanny, MaskedImage, ProcessedImage


# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[4]:


# ## Build a Lane Finding Pipeline
# 
# 

# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.
#os.listdir("test_images/")


for images in glob.iglob('test_images/*.jpg'):
    loadedImage = images
    image = mpimg.imread(loadedImage)
    print('Test Images: %s' % images)
    detect_lanes(image, fileCount)
    fileCount += 1



# In[5]:


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.


# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`
# 
# **Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**
#print('Left line: %s'%leftLineGradArray)
#print('Right Line: %s'%rightLineGradArray)

# In[ ]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[ ]:



def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    # Convert Image to Greyscale
    grey = grayscale(image)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    BlurredImage = gaussian_blur(grey, kernel_size)

    # Define parameters for Canny Edge Detection
    low_threshold = 50
    high_threshold = 150
    BlurredImageCanny = canny(BlurredImage, low_threshold, high_threshold)

    # Mask Edges of Defined Polygon using measurements based on image size for scaling
    imshape = image.shape
    vertices = np.array([[(int(round(imshape[1]/6.4)),imshape[0]),(int(round(imshape[1]/2.133)), int(round(imshape[0]/1.6875))), (int(round(imshape[1]/1.88)), int(round(imshape[0]/1.6875))), (int(round(imshape[1]/1.055)),imshape[0])]], dtype=np.int32)
    MaskedImage = region_of_interest(BlurredImageCanny, vertices)

    # Hough Transform Parameters
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = 0.027*np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 5     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10 #minimum number of pixels making up a line
    max_line_gap = 2    # maximum gap in pixels between connectable line segments
    DetectedLines = hough_lines(MaskedImage, rho, theta, threshold, min_line_length, max_line_gap)

    # Merge Images
    ProcessedImage = weighted_img(DetectedLines, image, α=0.8, β=1., λ=0.)

    return ProcessedImage


# Let's try the one with the solid white lane on the right first ...

# In[ ]:

count.clear()
xLeftAverageAppend = []
xRightAverageAppend = []
white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')
#print('Right Line average x array is : %s'%xRightAverageArray)


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[ ]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# ## Improve the draw_lines() function
# 
# **At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**
# 
# **Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[ ]:

count.clear()
xLeftAverageAppend = []
xRightAverageAppend = []
yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
get_ipython().run_line_magic('time', 'yellow_clip.write_videofile(yellow_output, audio=False)')


# In[ ]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


# ## Writeup and Submission
# 
# If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.
# 

# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[ ]:

count.clear()
xLeftAverageAppend = []
xRightAverageAppend = []
challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
get_ipython().run_line_magic('time', 'challenge_clip.write_videofile(challenge_output, audio=False)')


# In[ ]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))

