## Use official ROS Jazzy image as base
FROM ros:jazzy

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=jazzy

RUN apt-get update

RUN apt-get install -y \
    sudo \
    git \
    wget \
    curl 

# Add Gazebo repository
RUN wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# Update package lists
RUN apt-get update

# Install additional ROS development tools (some might already be in the base image)
RUN apt-get install -y \
    ros-dev-tools \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    locales \
    lsb-release \
    gnupg

# Install Gazebo Harmonic
RUN apt-get install -y \
    gz-harmonic

# Install GUI support packages for X11 forwarding
RUN apt-get install -y \
    x11-apps \
    xauth \
    xvfb

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -s /bin/bash developer && \
    usermod -aG sudo developer && \
    echo "developer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set up environment for the user
USER developer
WORKDIR /home/developer


# Create workspace
RUN mkdir -p /home/developer/ros2_ws/src

# Clone the specific repository
WORKDIR /home/developer/ros2_ws/src
RUN git clone https://github.com/fra-scarfato/line_tracking_race.git && \
    cd line_tracking_race 

RUN echo "export GZ_VERSION=harmonic ENVharmonic=harmonic" >> ~/.bashrc
ENV GZ_VERSION=harmonic

RUN sudo apt-get update

# Initialize rosdep (might already be done in base image, but safe to run)
RUN rosdep update
RUN sudo apt install -y pip

# Build the workspace
WORKDIR /home/developer/ros2_ws
RUN bash -c "source /opt/ros/jazzy/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -i -y --rosdistro jazzy && \
    colcon build --cmake-args -DBUILD_TESTING=ON"

# Set up ROS 2 environment in bashrc (ROS environment already sourced in base image)
RUN echo "source /home/developer/ros2_ws/install/setup.bash" >> ~/.bashrc

# Set environment variables for GUI
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1

# Default command
CMD ["/bin/bash"]


