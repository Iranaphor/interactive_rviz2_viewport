# ──────────────────────────────
# 1. Base – ROS 2 Humble
# ──────────────────────────────
FROM osrf/ros:humble-desktop-full

# before any 'apt-get update' that touches ros2-testing
RUN apt-get update && apt-get install -y --no-install-recommends gnupg2 \
 && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F42ED6FBAB17C654

 RUN echo "deb http://packages.ros.org/ros2-testing/ubuntu $(lsb_release -cs) main" \
 | tee /etc/apt/sources.list.d/ros2-testing.list

ARG  DEBIAN_FRONTEND=noninteractive
ARG  USERNAME=ros
ARG  UID=1000
ARG  GID=1000

# ──────────────────────────────
# 2. Non‑root user with sudo
# ──────────────────────────────


RUN apt-get update && apt-get install -y sudo git \
 && addgroup --gid ${GID} ${USERNAME} \
 && adduser  --uid ${UID} --gid ${GID} --disabled-password --gecos "" ${USERNAME} \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# ──────────────────────────────
# 3. Empty workspace skeleton
#    (sources come in as volumes at run‑time)
# ──────────────────────────────
ENV ROS_WS=/home/${USERNAME}/ros2_ws
RUN mkdir -p ${ROS_WS}/src \
 && chown -R ${USERNAME}:${USERNAME} ${ROS_WS}

# ──────────────────────────────
# 4. Toolchain + rosdep
#    (still running as root)
# ──────────────────────────────
# 1. Core build & ROS helpers
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo git build-essential \
        python3 python3-pip \
        python3-colcon-common-extensions \
        python3-rosdep python3-rosdep-modules && \
    rm -rf /var/lib/apt/lists/*

# 2. Convenience CLI tools      
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    nano tmux \   
    curl unzip \                                     
    htop net-tools tree && \
    rm -rf /var/lib/apt/lists/*

# 3. Extra ROS debs used by your workspace
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ros-humble-launch-testing \
        ros-humble-tf-transformations \ 
        ros-humble-launch-pytest \
        ros-humble-ament-cmake-core \
        && \
    rm -rf /var/lib/apt/lists/*

# 4. Python pip installs
RUN python3 -m pip install --no-cache-dir --upgrade \
    pip wheel \
    setuptools==58.2.0 \
    packaging==24.2 \
    tmule==1.5.9


# 5. Extra tools for visualisations
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        fonts-ubuntu \
        && \
    rm -rf /var/lib/apt/lists/*


# ──────────────────────────────
# 5. Drop privileges
# ──────────────────────────────
USER ${USERNAME}
WORKDIR ${ROS_WS}

# ──────────────────────────────
# 6. Friendly hook:
#    (source extension)
# ──────────────────────────────
RUN sed -i 's/^#\(force_color_prompt\)/\1/' /home/${USERNAME}/.bashrc

# ── Friendly hook: source your extension if it exists ───────────────
RUN echo '[ -f $HOME/bash_scripts/bashrc_extension.sh ] && \
source $HOME/bash_scripts/bashrc_extension.sh' \
      >> /home/${USERNAME}/.bashrc

# ──────────────────────────────
# 7. Default shell
# ──────────────────────────────
CMD ["/bin/bash"]
