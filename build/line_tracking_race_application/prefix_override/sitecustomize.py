import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/developer/ros2_ws/src/line_tracking_race/install/line_tracking_race_application'
