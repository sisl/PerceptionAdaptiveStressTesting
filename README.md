# SequentialPerceptionPipeline

## Setup
This project requires python 3.6 and pytorch 1.3. In a new `conda` environment, you can install these with:

```conda install python=3.6 pytorch=1.3 cudatoolkit=10.0 cudnn boost mayavi importlib_metadata importlib_resources```

And ensure:
>torch._C._GLIBCXX_USE_CXX11_ABI

is True

Next install the NuScenes python devkit. Follow the instructions here: https://github.com/nutonomy/nuscenes-devkit.

The nuscenes devkit also includes instructions on downloading the NuScenes dataset. To run this project, you will need the following data:
* [NuScenes v1.0-mini](https://github.com/nutonomy/nuscenes-devkit#nuscenes-setup)
* [Map expansion](https://github.com/nutonomy/nuscenes-devkit#map-expansion)


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

<!-- Now everything should be ready to run. Run the following
python ObjectDetect.py --data_path PATH/TO/KITTI/DATA/velodyne_points/data/ -->

<!-- This will create input images in a directory InputImages/ -->

<!-- # Trajectory prediction
After the input images are created they are in the format needed for trajectory prediction on covernet: 
https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/prediction_tutorial.ipynb
The link above is to a tutorial on how to predict trajectories. Covernet expects a map layered onto the input images of cars, however we do not have
that data so we have elected to create an empy map representation which will eliminate the map on nuscenes data.

For the agent state vector that information can be pulled from the oxts/ files in the ego vehicle. -->
