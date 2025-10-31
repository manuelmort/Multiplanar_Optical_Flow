#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    SetEnvironmentVariable,
    RegisterEventHandler,
    IncludeLaunchDescription
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # ───────────────────────────────
    # Launch Arguments
    # ───────────────────────────────
    declared_args = [
        DeclareLaunchArgument('x', default_value='-10.5'),
        DeclareLaunchArgument('y', default_value='-4.5'),
        DeclareLaunchArgument('z', default_value='0'),
        DeclareLaunchArgument('yaw', default_value='0'),
    ]

    pkg_share = FindPackageShare('optical_flow_proximity')
    xacro_path = PathJoinSubstitution([pkg_share, 'urdf', 'jackal_gazebo.urdf.xacro'])
    world_path = PathJoinSubstitution([pkg_share, 'GazeboWorlds', 'corridor_2.world'])
    rviz_config = PathJoinSubstitution([pkg_share, 'rviz', 'optical_flow_view.rviz'])

    robot_description = ParameterValue(Command(['xacro ', xacro_path]), value_type=str)

    # ───────────────────────────────
    # Environment setup for Gazebo
    # ───────────────────────────────
    set_gz_plugin_env = SetEnvironmentVariable(
        name='GZ_SIM_SYSTEM_PLUGIN_PATH',
        value='/opt/ros/jazzy/lib'
    )

    # ───────────────────────────────
    # Gazebo Simulation
    # ───────────────────────────────
    gz_world = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world_path],
        output='screen'
    )

    # ───────────────────────────────
    # Robot State Publisher
    # ───────────────────────────────
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}, {'use_sim_time': True}],
        output='screen'
    )

    # ───────────────────────────────
    # Spawn Jackal Entity
    # ───────────────────────────────
    spawn_entity = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros_gz_sim', 'create',
            '--name', 'jackal',
            '--x', LaunchConfiguration('x'),
            '--y', LaunchConfiguration('y'),
            '--z', LaunchConfiguration('z'),
            '--Y', LaunchConfiguration('yaw'),
            '--topic', 'robot_description'
        ],
        output='screen'
    )

    # ───────────────────────────────
    # Controller Spawners
    # ───────────────────────────────
    spawner_jsb = ExecuteProcess(
        cmd=['ros2', 'run', 'controller_manager', 'spawner', 'joint_state_broadcaster'],
        output='screen'
    )

    spawner_diffdrive = ExecuteProcess(
        cmd=['ros2', 'run', 'controller_manager', 'spawner', 'jackal_velocity_controller'],
        output='screen'
    )

    load_jsb = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[spawner_jsb],
        )
    )

    load_diffdrive = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawner_jsb,
            on_exit=[spawner_diffdrive],
        )
    )

    # ───────────────────────────────
    # ROS–Gazebo Bridge
    # ───────────────────────────────
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            "/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model",
            "/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V",
            "/camera/image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
        ],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    optical_flow_node = Node(
        package='vision_based_navigation_ttt',
        executable='optical_flow.py',
        name='optical_flow',
        arguments=['1'],
        parameters=[{'image_sub_name': '/camera/image'}, {'use_sim_time': True}],
        output='screen'
    )
    
    tau_node = Node(
        package='vision_based_navigation_ttt',
        executable='tau_computation.py',
        name='tau_computation',
        parameters=[{'image_sub_name': '/camera/image'}, {'use_sim_time': True}],
        output='screen'
    )

    controller_node = Node(
        package='vision_based_navigation_ttt',
        executable='controller.py',
        name='controller',
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    # ───────────────────────────────
    # Optical Flow Node
    # ───────────────────────────────
    flow_node = Node(
        package='optical_flow_proximity',
        executable='flow_proximity_node.py',
        name='flow_proximity_node',
        parameters=[{'image_topic': '/camera/image'}, {'use_sim_time': True}],
        output='screen'
    )

    # ───────────────────────────────
    # RViz2 Visualization
    # ───────────────────────────────
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    # ───────────────────────────────
    # Return launch description
    # ───────────────────────────────
    return LaunchDescription(declared_args + [
        set_gz_plugin_env,
        gz_world,
        robot_state_publisher,
        spawn_entity,
        load_jsb,
        load_diffdrive,
        bridge,
        flow_node,
        rviz,
        optical_flow_node, 
        tau_node,
        controller_node,
 
    ])

