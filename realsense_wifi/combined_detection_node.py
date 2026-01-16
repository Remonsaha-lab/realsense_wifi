#!/usr/bin/env python3
"""
Combined Detection Node
Runs both crater detection (best.pt) and object detection (best_object.pt) simultaneously
on the same RGB frames from RealSense camera via WiFi bridge.
Publishes combined detection results with different visual markers.
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
import threading

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# Color palette for different object classes
OBJECT_COLORS = [
    (255, 128, 0),   # Orange
    (255, 0, 128),   # Pink
    (128, 0, 255),   # Purple
    (0, 128, 255),   # Light Blue
    (128, 255, 0),   # Lime
    (255, 255, 0),   # Yellow
    (0, 255, 255),   # Cyan
    (255, 0, 255),   # Magenta
    (128, 128, 255), # Light Purple
    (255, 128, 128), # Light Red
]

# Crater detection color
CRATER_COLOR = (0, 255, 0)  # Green


class CombinedDetectionNode(Node):
    def __init__(self):
        super().__init__('combined_detection_node')
        
        # Declare parameters
        self.declare_parameter('crater_model_path', '/home/remon/Documents/crater_image_detect/best.pt')
        self.declare_parameter('object_model_path', '/home/remon/Documents/crater_image_detect/best_object.pt')
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('inference_size', 480)
        self.declare_parameter('enable_visualization', True)
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('enable_crater_detection', True)
        self.declare_parameter('enable_object_detection', True)
        
        # Get parameters
        self.crater_model_path = self.get_parameter('crater_model_path').value
        self.object_model_path = self.get_parameter('object_model_path').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.inference_size = self.get_parameter('inference_size').value
        self.enable_viz = self.get_parameter('enable_visualization').value
        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        self.enable_crater = self.get_parameter('enable_crater_detection').value
        self.enable_object = self.get_parameter('enable_object_detection').value
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Load YOLO models
        self.crater_model = None
        self.object_model = None
        self.object_class_names = {}
        
        if YOLO_AVAILABLE:
            self._load_models()
        else:
            self.get_logger().error('ultralytics not installed! Run: pip install ultralytics')
        
        # Frame storage with thread lock
        self.latest_rgb = None
        self.latest_depth = None
        self.frame_lock = threading.Lock()
        self.depth_scale = 0.001  # Default depth scale (mm to meters)
        
        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, rgb_topic, self._rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, depth_topic, self._depth_callback, 10)
        
        # Publishers
        self.detection_pub = self.create_publisher(
            Image, '/combined_detection/image', 10)
        self.crater_detection_pub = self.create_publisher(
            Image, '/crater_detection/image', 10)
        self.object_detection_pub = self.create_publisher(
            Image, '/object_detection/image', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/combined_detection/markers', 10)
        
        # Detection timer
        self.create_timer(0.033, self._detect_callback)  # ~30 FPS
        
        self.get_logger().info('Combined Detection Node initialized')
        self.get_logger().info(f'Crater Detection: {"ENABLED" if self.enable_crater else "DISABLED"}')
        self.get_logger().info(f'Object Detection: {"ENABLED" if self.enable_object else "DISABLED"}')
    
    def _load_models(self):
        """Load both YOLO models"""
        # Load Crater Model
        if self.enable_crater and os.path.exists(self.crater_model_path):
            try:
                self.crater_model = YOLO(self.crater_model_path)
                self.get_logger().info(f'Crater model loaded: {self.crater_model_path}')
            except Exception as e:
                self.get_logger().error(f'Failed to load crater model: {e}')
        elif self.enable_crater:
            self.get_logger().error(f'Crater model not found: {self.crater_model_path}')
        
        # Load Object Model
        if self.enable_object and os.path.exists(self.object_model_path):
            try:
                self.object_model = YOLO(self.object_model_path)
                if hasattr(self.object_model, 'names'):
                    self.object_class_names = self.object_model.names
                self.get_logger().info(f'Object model loaded: {self.object_model_path}')
                self.get_logger().info(f'Object classes: {self.object_class_names}')
            except Exception as e:
                self.get_logger().error(f'Failed to load object model: {e}')
        elif self.enable_object:
            self.get_logger().error(f'Object model not found: {self.object_model_path}')
    
    def _rgb_callback(self, msg):
        """Store latest RGB frame"""
        try:
            with self.frame_lock:
                self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f'RGB conversion error: {e}')
    
    def _depth_callback(self, msg):
        """Store latest depth frame"""
        try:
            with self.frame_lock:
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except Exception as e:
            self.get_logger().warn(f'Depth conversion error: {e}')
    
    def _get_object_color(self, class_id):
        """Get color for an object class ID"""
        return OBJECT_COLORS[class_id % len(OBJECT_COLORS)]
    
    def _get_distance(self, center_x, center_y):
        """Get distance from depth frame at given pixel"""
        if self.latest_depth is None:
            return 0.0
        
        h, w = self.latest_depth.shape[:2]
        if 0 <= center_x < w and 0 <= center_y < h:
            depth_value = self.latest_depth[center_y, center_x]
            return float(depth_value) * self.depth_scale
        return 0.0
    
    def _detect_callback(self):
        """Run both detections on latest frame"""
        with self.frame_lock:
            if self.latest_rgb is None:
                return
            rgb_frame = self.latest_rgb.copy()
            depth_frame = self.latest_depth.copy() if self.latest_depth is not None else None
        
        # Create annotated frames
        combined_frame = rgb_frame.copy()
        crater_frame = rgb_frame.copy()
        object_frame = rgb_frame.copy()
        
        markers = MarkerArray()
        marker_id = 0
        stamp = self.get_clock().now().to_msg()
        
        # =====================================================
        # CRATER DETECTION
        # =====================================================
        if self.enable_crater and self.crater_model is not None:
            try:
                crater_results = self.crater_model.predict(
                    rgb_frame, 
                    conf=self.conf_threshold, 
                    imgsz=self.inference_size, 
                    verbose=False
                )
                
                for result in crater_results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        distance = self._get_distance(center_x, center_y)
                        
                        # Draw on frames
                        for frame in [combined_frame, crater_frame]:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), CRATER_COLOR, 3)
                            label = f'CRATER {conf:.0%} | {distance:.2f}m'
                            
                            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), CRATER_COLOR, -1)
                            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                            cv2.circle(frame, (center_x, center_y), 6, (0, 0, 255), -1)
                        
                        # Create sphere marker for crater
                        marker = Marker()
                        marker.header.frame_id = 'camera_color_optical_frame'
                        marker.header.stamp = stamp
                        marker.ns = 'crater_detections'
                        marker.id = marker_id
                        marker.type = Marker.SPHERE
                        marker.action = Marker.ADD
                        
                        marker.pose.position.x = float(center_x - rgb_frame.shape[1]/2) * distance / 600.0
                        marker.pose.position.y = float(center_y - rgb_frame.shape[0]/2) * distance / 600.0
                        marker.pose.position.z = distance
                        marker.pose.orientation.w = 1.0
                        
                        marker.scale.x = 0.4
                        marker.scale.y = 0.4
                        marker.scale.z = 0.2
                        
                        marker.color.r = 0.0
                        marker.color.g = 1.0
                        marker.color.b = 0.0
                        marker.color.a = 0.8
                        
                        marker.lifetime.sec = 0
                        marker.lifetime.nanosec = 500000000
                        markers.markers.append(marker)
                        
                        # Text marker
                        text_marker = Marker()
                        text_marker.header = marker.header
                        text_marker.ns = 'crater_labels'
                        text_marker.id = marker_id + 5000
                        text_marker.type = Marker.TEXT_VIEW_FACING
                        text_marker.action = Marker.ADD
                        text_marker.pose.position.x = marker.pose.position.x
                        text_marker.pose.position.y = marker.pose.position.y - 0.25
                        text_marker.pose.position.z = marker.pose.position.z
                        text_marker.scale.z = 0.15
                        text_marker.color.r = 0.0
                        text_marker.color.g = 1.0
                        text_marker.color.b = 0.0
                        text_marker.color.a = 1.0
                        text_marker.text = f'CRATER ({distance:.1f}m)'
                        text_marker.lifetime.sec = 0
                        text_marker.lifetime.nanosec = 500000000
                        markers.markers.append(text_marker)
                        
                        marker_id += 1
                        
                        self.get_logger().debug(
                            f'[CRATER] conf={conf:.2%}, dist={distance:.2f}m, pos=({center_x}, {center_y})'
                        )
                        
            except Exception as e:
                self.get_logger().error(f'Crater detection error: {e}')
        
        # =====================================================
        # OBJECT DETECTION
        # =====================================================
        if self.enable_object and self.object_model is not None:
            try:
                object_results = self.object_model.predict(
                    rgb_frame, 
                    conf=self.conf_threshold, 
                    imgsz=self.inference_size, 
                    verbose=False
                )
                
                for result in object_results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        
                        class_name = self.object_class_names.get(cls, f'Object_{cls}') if isinstance(self.object_class_names, dict) else f'Object_{cls}'
                        color = self._get_object_color(cls)
                        
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        distance = self._get_distance(center_x, center_y)
                        
                        # Draw on frames
                        for frame in [combined_frame, object_frame]:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            label = f'{class_name} {conf:.0%} | {distance:.2f}m'
                            
                            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1), color, -1)
                            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
                        
                        # Create cube marker for object
                        marker = Marker()
                        marker.header.frame_id = 'camera_color_optical_frame'
                        marker.header.stamp = stamp
                        marker.ns = 'object_detections'
                        marker.id = marker_id
                        marker.type = Marker.CUBE
                        marker.action = Marker.ADD
                        
                        marker.pose.position.x = float(center_x - rgb_frame.shape[1]/2) * distance / 600.0
                        marker.pose.position.y = float(center_y - rgb_frame.shape[0]/2) * distance / 600.0
                        marker.pose.position.z = distance
                        marker.pose.orientation.w = 1.0
                        
                        box_width = abs(x2 - x1)
                        box_height = abs(y2 - y1)
                        marker.scale.x = max(0.1, box_width * distance / 600.0)
                        marker.scale.y = max(0.1, box_height * distance / 600.0)
                        marker.scale.z = 0.1
                        
                        marker.color.r = color[2] / 255.0
                        marker.color.g = color[1] / 255.0
                        marker.color.b = color[0] / 255.0
                        marker.color.a = 0.7
                        
                        marker.lifetime.sec = 0
                        marker.lifetime.nanosec = 500000000
                        markers.markers.append(marker)
                        
                        # Text marker
                        text_marker = Marker()
                        text_marker.header = marker.header
                        text_marker.ns = 'object_labels'
                        text_marker.id = marker_id + 10000
                        text_marker.type = Marker.TEXT_VIEW_FACING
                        text_marker.action = Marker.ADD
                        text_marker.pose.position.x = marker.pose.position.x
                        text_marker.pose.position.y = marker.pose.position.y - 0.2
                        text_marker.pose.position.z = marker.pose.position.z
                        text_marker.scale.z = 0.12
                        text_marker.color.r = 1.0
                        text_marker.color.g = 1.0
                        text_marker.color.b = 1.0
                        text_marker.color.a = 1.0
                        text_marker.text = f'{class_name} ({distance:.1f}m)'
                        text_marker.lifetime.sec = 0
                        text_marker.lifetime.nanosec = 500000000
                        markers.markers.append(text_marker)
                        
                        marker_id += 1
                        
                        self.get_logger().debug(
                            f'[OBJECT] {class_name}, conf={conf:.2%}, dist={distance:.2f}m'
                        )
                        
            except Exception as e:
                self.get_logger().error(f'Object detection error: {e}')
        
        # =====================================================
        # PUBLISH RESULTS
        # =====================================================
        if self.enable_viz:
            # Publish combined detection image
            combined_msg = self.bridge.cv2_to_imgmsg(combined_frame, 'bgr8')
            combined_msg.header.stamp = stamp
            combined_msg.header.frame_id = 'camera_color_optical_frame'
            self.detection_pub.publish(combined_msg)
            
            # Publish individual detection images
            if self.enable_crater:
                crater_msg = self.bridge.cv2_to_imgmsg(crater_frame, 'bgr8')
                crater_msg.header = combined_msg.header
                self.crater_detection_pub.publish(crater_msg)
            
            if self.enable_object:
                object_msg = self.bridge.cv2_to_imgmsg(object_frame, 'bgr8')
                object_msg.header = combined_msg.header
                self.object_detection_pub.publish(object_msg)
        
        # Publish markers
        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = CombinedDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
