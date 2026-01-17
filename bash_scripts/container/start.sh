#!/usr/bin/env bash
# ------------------------------------------------------------
# First-run setup script for ROS 2 development container
# ------------------------------------------------------------

MARKER_FILE="$HOME/.ros2_first_run_done"
ls -a $HOME

echo -e "\n\n\nSourcing /opt/ros/humble/setup.bash\n"
source /opt/ros/humble/setup.bash


echo -e "\n\n\nSet working directory to /home/ros/ros2_ws\n"
[ -d "$HOME/ros2_ws" ] && cd "$HOME/ros2_ws"


if [ ! -f "$MARKER_FILE" ]; then

    echo -e "\n\n\nRunning apt update\n"
    sudo apt-get update


    echo -e "\n\n\nRunning ROSDep update and install\n"
    rosdep update
    rosdep install --from-paths src --ignore-src -r -y


    echo -e "\n\n\nBuilding Colcon Workspace at /home/ros/ros2_ws\n"
    colcon build --symlink-install


    echo -e "\n\n\nMarking first run complete\n"
    touch "$MARKER_FILE"
else

    echo -e "\n\n\nFirst run already completed — skipping setup steps\n"
fi


echo -e "\n\n\nSourcing built workspace\n"
[ -f "$HOME/ros2_ws/install/setup.bash" ] && source "$HOME/ros2_ws/install/setup.bash"


echo -e "\n\n\nCustomise ros logger\n"
export RCUTILS_CONSOLE_OUTPUT_FORMAT="{severity}: {message}"
export RCUTILS_COLORIZED_OUTPUT=1


echo -e "\n\n\nLaunching TMuLE (conditional)\n"
TMULE="$HOME/ros2_ws/src/rviz2_viewport_control/tmule/run.tmule.yaml"
if [ "$AUTO_LAUNCH_TMULE" = "true" ]; then
    tmule -c $TMULE launch
fi


echo -e "\n\n\n######################################\nStart script completed.\nContainer will continue in background.\n######################################\nOpen a new terminal and connect with:\n\n            rviz2_attach\n\n######################################\n\n\n\n"
exec sleep infinity
