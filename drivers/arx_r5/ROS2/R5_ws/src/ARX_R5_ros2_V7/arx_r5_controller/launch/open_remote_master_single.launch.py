from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    arm_name = LaunchConfiguration('arm_name')
    arm_can_id = LaunchConfiguration('arm_can_id')
    publish_rate_hz = LaunchConfiguration('remote_publish_rate_hz')
    qos_depth = LaunchConfiguration('remote_qos_depth')

    return LaunchDescription([
        DeclareLaunchArgument('arm_name', default_value='arm_master_l'),
        DeclareLaunchArgument('arm_can_id', default_value='can0'),
        DeclareLaunchArgument('remote_publish_rate_hz', default_value='200'),
        DeclareLaunchArgument('remote_qos_depth', default_value='1'),
        Node(
            package='arx_r5_controller',
            executable='R5Controller',
            name=arm_name,
            output='screen',
            parameters=[{
                'arm_can_id': arm_can_id,
                'arm_control_type': 'remote_master',
                'arm_pub_topic_name': 'arm_master_l_status',
                'arm_end_type': 1,
                'remote_publish_rate_hz': ParameterValue(publish_rate_hz, value_type=int),
                'remote_qos_depth': ParameterValue(qos_depth, value_type=int),
            }],
        ),
    ])
