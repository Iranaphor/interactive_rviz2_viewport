#!/usr/bin/env bash
# ------------------------------------------------------------
# Custom environment for AHNDAS dev‑container
# ------------------------------------------------------------


# 1. ROS 2 distro environment
source /opt/ros/humble/setup.bash


# 2. Workspace overlay (if already built)
[ -f "$HOME/ros2_ws/install/setup.bash" ] && source "$HOME/ros2_ws/install/setup.bash"


# 3. ROS Configuration
export RCUTILS_CONSOLE_OUTPUT_FORMAT="{severity}: {message}"
export RCUTILS_COLORIZED_OUTPUT=1
export ROS_LOCALHOST_ONLY=0
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID}
echo "ROS Domain ID - $ROS_DOMAIN_ID (local only = $ROS_LOCALHOST_ONLY)"


# 4. Custom commands
alias t='tmux'
alias gs='git status'
alias nano='nano -BEPOSUWx -T 4'


# 5. Setup .tmule.conf
TMUX_CONF="$HOME/bash_scripts/tmux.conf"
[ ! -f "$HOME/.tmux.conf" ] && cp $TMUX_CONF "$HOME/.tmux.conf"
