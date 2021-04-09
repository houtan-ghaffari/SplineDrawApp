# SplineDrawApp
This repository contains a simple application for drawing splines, which is based on OpenCV and Scipy. You can extend this for your use cases if you find it helpful.

# Guide
- There is a folder called input_images. First, you need to put an image inside that folder, which you like to draw a spline on it. Please use .jpg files.
- Run the spline.py file inside the create_spline folder. It will ask you to enter the name of the image you put inside the input images folder, e.g., eagle.jpg. Then, a window will show up, and you can select the points you want by left-clicking on the image. For example, select some points along a curve in the image. After you selected the points, press the escape button on the keyboard to close the window. The program will save the coordinates that you selected along with the tck parameters for x-axis and y-axis separately. It will also plot the spline on the image and show it to you. The output files get saved in the outputs folder with the same name as your image. There is a sample output in the outputs folder.
- If you have already run the previous program, you have the tck parameters for drawing a spline. There is a file called visualize_tck.py inside the visualize_spline folder. You can run this script and give it the name of an image that you previously processed by the spline.py, e.g., eagle.jpg. It will draw the spline again based on its tck parameters.


