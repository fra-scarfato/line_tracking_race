cmake_minimum_required(VERSION 3.5)

project(line_tracking_race_application)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Install Python modules
install(
  DIRECTORY src/line_tracking
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files (if any)
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()
