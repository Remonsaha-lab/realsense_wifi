from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'realsense_wifi'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='remon',
    maintainer_email='remon@todo.com',
    description='RealSense WiFi streaming package for laptop receiver',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'combined_detection_node = realsense_wifi.combined_detection_node:main',
            'crater_detection_node = realsense_wifi.crater_detection_node:main',
            'object_detection_node = realsense_wifi.object_detection_node:main',
            'camera_receiver_node = realsense_wifi.camera_receiver_node:main',
        ],
    },
)
