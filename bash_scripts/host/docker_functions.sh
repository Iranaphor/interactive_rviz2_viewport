# Attach to the running dev container
function rviz2_attach () {
    cd $HOME/docker_workspaces/interactive_rviz2_viewport/docker
    xhost +local:docker
    docker compose exec ros2_interactive_rviz2_viewport bash -l
}
