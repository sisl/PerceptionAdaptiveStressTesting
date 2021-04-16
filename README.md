# SequentialPerceptionPipeline
Download pytorch using commands:
Conda install python-3.6 pytorch=1.1 cudatoolkit=10.0 cudnn boost mayavi importlib_metadata importlib_resources

And ensure:
>>torch._C._GLIBCXX_USE_CXX11_ABI
is True

Follow the instructions here: https://github.com/nutonomy/nuscenes-devkit to download the nuscenes devkit package

You should not need nuImages or the nuScenes data to continue with this project but you can download the nuscenes mini dataset 
along with the map expansion v1.3 to learn more about it. Or if you want to ever run the tutorial on the nuscenes-devkit for prediction,
you'll need the mini and map expansion data.

Now follow the installation instructions for https://github.com/traveller59/spconv
NOTE:
I had to add to the CMakeLists.txt at the top of the file
	set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.0/bin/nvcc) 
And add this to setup.py:
 '-DCMAKE_CUDA_COMPILER=/usr/local/cuda-10.0/bin/nvcc', 
right after the line with '-DPYBIND11_PYTHON_VERSION={}'.format(PYTHON_VERSION)

Now follow the installation instructions for: https://github.com/open-mmlab/OpenPCDet/tree/0642cf06d0fd84f50cc4c6c01ea28edbc72ea810

and download the PV-RCNN_8369.pth pre-trained model located: https://drive.google.com/file/d/1lIOq4Hxr0W3qsX83ilQv0nk1Cls6KAr-/view

Make a directory within OpenPCDet and save the model to "OpenPCDet/Models/pv_rcnn8369.pth"

Now everything should be ready to run. Run the following
python ObjectDetect.py --data_path PATH/TO/KITTI/DATA/velodyne_points/data/

This will create input images in a directory InputImages/
