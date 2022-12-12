# Tool Tracking on Taurus #

Dataset: https://drive.google.com/drive/folders/1Twt8h1oPO2eixuUngoWw0zGMWQPZUoa3?usp=sharing


## Documentation ##

`taurus_tool_tracking.ipynb`: read data from rosbag and perform tool tracking with DNN keypoint detector.

`taurus_tool_tracking_with_marker.ipynb`: read data from rosbag and perform tool tracking using painted markers.

`particle_filter.py`: implements particle filter with resample strategies, noise model, update and observation model.

`robot_fk.py`: implements robot forward kinematics.

`robot_fk_new.py`: inherit robot forward kinematic model from "fk_funcions.py".

`camera.py`: stores camera parameters and some drawing functions.

`point_feature.json`: defines the point feature with their postion w.r.t joints.

`point_feature_markers.json`: defines the point feature (markers) with their postion w.r.t joints. See "key_markers.png" for illustration.


## Keypoint detection ##

The point features are detected using the [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut). DLC provides GUI for labelling, training, and running inference for the point detection neural network. A ROS wrapper for running online DLC inference can be found [here](https://github.com/jingpeilu/DLC_ros).
