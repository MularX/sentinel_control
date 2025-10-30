from launch import LaunchDescription
from launch.actions import RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node

def generate_launch_description():
    navigate = Node(
        package='sentinel_control',
        executable='navigate_bt',
        name='navigate',
        output='screen'
    )

    path_planner = Node(
        package='sentinel_control',
        executable='path_planner_single',
        name='path_planner',
        output='screen'
    )

    # map = Node(
    #     package='sentinel_control',
    #     executable='map_obstacles',
    #     name='map',
    #     output='screen'
    # )

    path_planner_after_navigate = RegisterEventHandler(
        OnProcessStart(
            target_action=navigate,
            on_start=[path_planner],
        )
    )


    return LaunchDescription([
        navigate,
        path_planner_after_navigate,
    ])
