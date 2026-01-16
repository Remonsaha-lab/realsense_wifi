#!/usr/bin/env python3
"""
RealSense WiFi Camera Receiver Node
Receives camera frames from RealSense camera via WiFi bridge (Ubiquiti LiteBeam M5)
and publishes RGB and Depth images as ROS2 topics.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import socket
import struct
import cv2
import threading
from std_msgs.msg import Header


class CameraReceiverNode(Node):
    def __init__(self):
        super().__init__('camera_receiver_node')
        
        # Declare parameters
        self.declare_parameter('receiver_ip', '0.0.0.0')
        self.declare_parameter('rgb_port', 5000)
        self.declare_parameter('depth_port', 5001)
        self.declare_parameter('frame_width', 848)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('frame_rate', 30)
        self.declare_parameter('buffer_size', 65536)
        
        # Get parameters
        self.receiver_ip = self.get_parameter('receiver_ip').value
        self.rgb_port = self.get_parameter('rgb_port').value
        self.depth_port = self.get_parameter('depth_port').value
        self.frame_width = self.get_parameter('frame_width').value
        self.frame_height = self.get_parameter('frame_height').value
        self.frame_rate = self.get_parameter('frame_rate').value
        self.buffer_size = self.get_parameter('buffer_size').value
        
        # CV Bridge for ROS image conversion
        self.bridge = CvBridge()
        
        # Publishers
        self.rgb_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.rgb_info_pub = self.create_publisher(CameraInfo, '/camera/color/camera_info', 10)
        self.depth_info_pub = self.create_publisher(CameraInfo, '/camera/depth/camera_info', 10)
        
        # Camera info (default D435 intrinsics for 848x480)
        self.camera_info = self._create_camera_info()
        
        # Socket connections
        self.rgb_socket = None
        self.depth_socket = None
        self.running = True
        
        # Frame storage
        self.latest_rgb_frame = None
        self.latest_depth_frame = None
        self.frame_lock = threading.Lock()
        
        # Start receiver threads
        self.rgb_thread = threading.Thread(target=self._receive_rgb, daemon=True)
        self.depth_thread = threading.Thread(target=self._receive_depth, daemon=True)
        
        self._setup_sockets()
        self.rgb_thread.start()
        self.depth_thread.start()
        
        # Publisher timer
        self.create_timer(1.0 / self.frame_rate, self._publish_frames)
        
        self.get_logger().info(f'Camera Receiver Node started - Listening on RGB:{self.rgb_port}, Depth:{self.depth_port}')
    
    def _create_camera_info(self):
        """Create camera info message with D435 default intrinsics"""
        info = CameraInfo()
        info.width = self.frame_width
        info.height = self.frame_height
        info.distortion_model = 'plumb_bob'
        
        # Default D435 intrinsics (approximation for 848x480)
        fx = 616.0
        fy = 616.0
        cx = self.frame_width / 2.0
        cy = self.frame_height / 2.0
        
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        
        return info
    
    def _setup_sockets(self):
        """Setup UDP sockets for receiving camera data"""
        try:
            # RGB socket
            self.rgb_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.rgb_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.rgb_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.buffer_size * 4)
            self.rgb_socket.bind((self.receiver_ip, self.rgb_port))
            self.rgb_socket.settimeout(1.0)
            
            # Depth socket
            self.depth_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.depth_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.depth_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.buffer_size * 4)
            self.depth_socket.bind((self.receiver_ip, self.depth_port))
            self.depth_socket.settimeout(1.0)
            
            self.get_logger().info('UDP sockets initialized successfully')
            
        except Exception as e:
            self.get_logger().error(f'Failed to setup sockets: {e}')
    
    def _receive_rgb(self):
        """Thread to receive RGB frames"""
        frame_buffer = b''
        expected_size = 0
        
        while self.running:
            try:
                data, addr = self.rgb_socket.recvfrom(self.buffer_size)
                
                if len(data) < 8:
                    continue
                
                # Parse header: frame_id (4 bytes) + total_size (4 bytes)
                header = data[:8]
                frame_id, total_size = struct.unpack('!II', header)
                payload = data[8:]
                
                if expected_size == 0:
                    expected_size = total_size
                    frame_buffer = b''
                
                frame_buffer += payload
                
                if len(frame_buffer) >= expected_size:
                    # Decode JPEG frame
                    try:
                        nparr = np.frombuffer(frame_buffer[:expected_size], np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            with self.frame_lock:
                                self.latest_rgb_frame = frame
                    except Exception as e:
                        self.get_logger().warn(f'RGB decode error: {e}')
                    
                    frame_buffer = b''
                    expected_size = 0
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    self.get_logger().warn(f'RGB receive error: {e}')
    
    def _receive_depth(self):
        """Thread to receive Depth frames"""
        frame_buffer = b''
        expected_size = 0
        
        while self.running:
            try:
                data, addr = self.depth_socket.recvfrom(self.buffer_size)
                
                if len(data) < 8:
                    continue
                
                # Parse header
                header = data[:8]
                frame_id, total_size = struct.unpack('!II', header)
                payload = data[8:]
                
                if expected_size == 0:
                    expected_size = total_size
                    frame_buffer = b''
                
                frame_buffer += payload
                
                if len(frame_buffer) >= expected_size:
                    # Decode depth frame (PNG or raw)
                    try:
                        nparr = np.frombuffer(frame_buffer[:expected_size], np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                        
                        if frame is not None:
                            with self.frame_lock:
                                self.latest_depth_frame = frame
                    except Exception as e:
                        self.get_logger().warn(f'Depth decode error: {e}')
                    
                    frame_buffer = b''
                    expected_size = 0
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    self.get_logger().warn(f'Depth receive error: {e}')
    
    def _publish_frames(self):
        """Publish frames to ROS2 topics"""
        stamp = self.get_clock().now().to_msg()
        
        with self.frame_lock:
            rgb_frame = self.latest_rgb_frame
            depth_frame = self.latest_depth_frame
        
        # Publish RGB
        if rgb_frame is not None:
            try:
                rgb_msg = self.bridge.cv2_to_imgmsg(rgb_frame, 'bgr8')
                rgb_msg.header.stamp = stamp
                rgb_msg.header.frame_id = 'camera_color_optical_frame'
                self.rgb_pub.publish(rgb_msg)
                
                # Publish camera info
                self.camera_info.header.stamp = stamp
                self.camera_info.header.frame_id = 'camera_color_optical_frame'
                self.rgb_info_pub.publish(self.camera_info)
                
            except Exception as e:
                self.get_logger().warn(f'RGB publish error: {e}')
        
        # Publish Depth
        if depth_frame is not None:
            try:
                if len(depth_frame.shape) == 2:
                    depth_msg = self.bridge.cv2_to_imgmsg(depth_frame, '16UC1')
                else:
                    depth_msg = self.bridge.cv2_to_imgmsg(depth_frame, 'passthrough')
                
                depth_msg.header.stamp = stamp
                depth_msg.header.frame_id = 'camera_depth_optical_frame'
                self.depth_pub.publish(depth_msg)
                
                # Publish depth camera info
                self.camera_info.header.frame_id = 'camera_depth_optical_frame'
                self.depth_info_pub.publish(self.camera_info)
                
            except Exception as e:
                self.get_logger().warn(f'Depth publish error: {e}')
    
    def destroy_node(self):
        """Cleanup on shutdown"""
        self.running = False
        
        if self.rgb_socket:
            self.rgb_socket.close()
        if self.depth_socket:
            self.depth_socket.close()
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraReceiverNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
