#!/usr/bin/env python3
"""
Object Detection Node
Uses YOLO model (best_object.pt) to detect general objects in RGB images from RealSense camera.
Publishes detection results and annotated images.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
import cv2
import os

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# Color palette for different object classes
CLASS_COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
    (0, 128, 128),  # Teal
    (128, 128, 0),  # Olive
]


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        
        # Declare parameters
        self.declare_parameter('model_path', '/home/remon/Documents/crater_image_detect/best_object.pt')
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('inference_size', 480)
        self.declare_parameter('enable_visualization', True)
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        
        # Get parameters
        self.model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.inference_size = self.get_parameter('inference_size').value
        self.enable_viz = self.get_parameter('enable_visualization').value
        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Load YOLO model
        self.model = None
        self.class_names = []
        if YOLO_AVAILABLE:
            self._load_model()
        else:
            self.get_logger().error('ultralytics not installed! Run: pip install ultralytics')
        
        # Frame storage
        self.latest_rgb = None
        self.latest_depth = None
        self.depth_scale = 0.001  # Default depth scale (mm to meters)
        
        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, rgb_topic, self._rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, depth_topic, self._depth_callback, 10)
        
        # Publishers
        self.detection_pub = self.create_publisher(
            Image, '/object_detection/image', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/object_detection/markers', 10)
        
        # Detection timer
        self.create_timer(0.033, self._detect_callback)  # ~30 FPS
        
        self.get_logger().info('Object Detection Node initialized')
    
    def _load_model(self):
        """Load YOLO model"""
        if os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                # Get class names from model
                if hasattr(self.model, 'names'):
                    self.class_names = self.model.names
                self.get_logger().info(f'Object model loaded: {self.model_path}')
                self.get_logger().info(f'Classes: {self.class_names}')
            except Exception as e:
                self.get_logger().error(f'Failed to load model: {e}')
        else:
            self.get_logger().error(f'Model not found: {self.model_path}')
    
    def _rgb_callback(self, msg):
        """Store latest RGB frame"""
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f'RGB conversion error: {e}')
    
    def _depth_callback(self, msg):
        """Store latest depth frame"""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except Exception as e:
            self.get_logger().warn(f'Depth conversion error: {e}')
    
    def _get_color(self, class_id):
        """Get color for a class ID"""
        return CLASS_COLORS[class_id % len(CLASS_COLORS)]
    
    def _detect_callback(self):
        """Run detection on latest frame"""
        if self.model is None or self.latest_rgb is None:
            return
        
        try:
            # Run YOLO inference
            results = self.model.predict(
                self.latest_rgb, 
                conf=self.conf_threshold, 
                imgsz=self.inference_size, 
                verbose=False
            )
            
            # Process results
            annotated_frame = self.latest_rgb.copy()
            markers = MarkerArray()
            marker_id = 0
            
            for result in results:
                for box in result.boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Get class name
                    class_name = self.class_names.get(cls, f'Class_{cls}') if isinstance(self.class_names, dict) else f'Object_{cls}'
                    
                    # Get color for this class
                    color = self._get_color(cls)
                    
                    # Calculate center
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Get distance from depth
                    distance = 0.0
                    if self.latest_depth is not None:
                        h, w = self.latest_depth.shape[:2]
                        if 0 <= center_x < w and 0 <= center_y < h:
                            depth_value = self.latest_depth[center_y, center_x]
                            distance = float(depth_value) * self.depth_scale
                    
                    # Draw on annotated frame
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    label = f'{class_name} {conf:.0%} | {distance:.2f}m'
                    
                    # Background for text
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    
                    # Create marker for RViz
                    marker = Marker()
                    marker.header.frame_id = 'camera_color_optical_frame'
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = 'object_detections'
                    marker.id = marker_id
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD
                    
                    # Position in camera frame (simplified)
                    marker.pose.position.x = float(center_x - annotated_frame.shape[1]/2) * distance / 600.0
                    marker.pose.position.y = float(center_y - annotated_frame.shape[0]/2) * distance / 600.0
                    marker.pose.position.z = distance
                    marker.pose.orientation.w = 1.0
                    
                    # Scale based on bounding box size
                    box_width = abs(x2 - x1)
                    box_height = abs(y2 - y1)
                    marker.scale.x = max(0.1, box_width * distance / 600.0)
                    marker.scale.y = max(0.1, box_height * distance / 600.0)
                    marker.scale.z = 0.1
                    
                    # Color from class
                    marker.color.r = color[2] / 255.0
                    marker.color.g = color[1] / 255.0
                    marker.color.b = color[0] / 255.0
                    marker.color.a = 0.7
                    
                    marker.lifetime.sec = 0
                    marker.lifetime.nanosec = 500000000  # 0.5 seconds
                    
                    markers.markers.append(marker)
                    
                    # Add text marker for class name
                    text_marker = Marker()
                    text_marker.header = marker.header
                    text_marker.ns = 'object_labels'
                    text_marker.id = marker_id + 1000
                    text_marker.type = Marker.TEXT_VIEW_FACING
                    text_marker.action = Marker.ADD
                    text_marker.pose.position.x = marker.pose.position.x
                    text_marker.pose.position.y = marker.pose.position.y - 0.2
                    text_marker.pose.position.z = marker.pose.position.z
                    text_marker.scale.z = 0.15
                    text_marker.color.r = 1.0
                    text_marker.color.g = 1.0
                    text_marker.color.b = 1.0
                    text_marker.color.a = 1.0
                    text_marker.text = f'{class_name} ({distance:.1f}m)'
                    text_marker.lifetime.sec = 0
                    text_marker.lifetime.nanosec = 500000000
                    markers.markers.append(text_marker)
                    
                    marker_id += 1
                    
                    # Log detection
                    self.get_logger().info(
                        f'Object detected: {class_name}, conf={conf:.2%}, dist={distance:.2f}m, '
                        f'pos=({center_x}, {center_y})'
                    )
            
            # Publish annotated image
            if self.enable_viz:
                img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, 'bgr8')
                img_msg.header.stamp = self.get_clock().now().to_msg()
                img_msg.header.frame_id = 'camera_color_optical_frame'
                self.detection_pub.publish(img_msg)
            
            # Publish markers
            self.marker_pub.publish(markers)
            
        except Exception as e:
            self.get_logger().error(f'Detection error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
