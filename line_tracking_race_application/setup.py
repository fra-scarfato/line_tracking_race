from setuptools import setup
import os
from glob import glob

package_name = 'line_tracking_race_application'

setup(
    name=package_name,
    version='0.1.0',
    packages=[
        'line_tracking',
        'line_tracking.nodes',
        'line_tracking.planning_strategies',
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files here
        ('share/' + package_name + '/launch', glob('launch/*.py')),  # or *.launch.py if that's your extension
        # If you have config files, also include:
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    package_dir={'': 'src'},
    install_requires=['setuptools', 'pynput'],  # include external dependencies here
    zip_safe=False,
    maintainer='Me',
    maintainer_email='me@email.com',
    description='Line tracking logic for a differential-drive robot in ROS 2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'planner_node = line_tracking.nodes.planner_node:main',
            'control_node = line_tracking.nodes.control_node:main',
            'referee_node = line_tracking.nodes.referee_node:main',
        ],
    },
)
