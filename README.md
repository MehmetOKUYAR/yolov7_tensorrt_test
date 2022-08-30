# Yolov7 TensorRt Traffic Sign Detection
 
You can create your own lane segmentation model using **this dataset** https://www.kaggle.com/datasets/mehmetokuyar/traffic-sign-dataset , then you can set up a traffic sign detection algorithm by referencing this code.

**Note:** You can easily use this code in your own projects by changing the class names in the code.

### Run the application
The input parameters can be changed using the command line :
~~~
yolo_aug.py -i <input_dir> -w <weight_path> -o <output_dir>
~~~~~~~~~
For running image test example :
~~~~
python3 yolov7_tensorrt_test.py -i path/image.jpg -w path/yolov7.trt -o path/
~~~~~~~~~

For running video test example :
~~~~
python3 yolov7_tensorrt_test.py -v path/video.mp4 -w path/yolov7.trt -o path/
~~~~~~~~~
# Example Output

<a href="https://youtu.be/Xp0j-O2TiRE" title="Traffic Sign Detection">
  <p align="center">
    <img width="75%" src="https://github.com/MehmetOKUYAR/yolov7_tensorrt_test/blob/main/results/result.jpg" alt="Traffic Sign Detection"/>
  </p>
</a>

