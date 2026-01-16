from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([
        # RealSense Camera Node
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='camera',
            output='screen',
            parameters=[
                {
                    "pointcloud.enable": False,
                    "align_depth.enable": True,
                    "rgb_camera.profile": "640x480x30",
                    "depth_module.profile": "640x480x30"
                },
                "/home/obseract/ros2_ws/src/realsense_wifi/config/realsense_params.yaml"
            ]
        ),
        
        # Combined Detection Node (crater + object detection)
        Node(
            package='realsense_wifi',
            executable='combined_detection_node',
            name='combined_detection',
            output='screen',
            parameters=[
                {
                    "crater_model_path": "/home/obseract/Documents/crater_image_detect/best.pt",
                    "object_model_path": "/home/obseract/Documents/crater_image_detect/best_object.pt",
                    "confidence_threshold": 0.25,
                    "inference_size": 480,
                    "enable_visualization": True,
                    "rgb_topic": "/camera/camera/color/image_raw",
                    "depth_topic": "/camera/camera/depth/image_rect_raw",
                    "enable_crater_detection": True,
                    "enable_object_detection": True
                }
            ]
        )
    ])
