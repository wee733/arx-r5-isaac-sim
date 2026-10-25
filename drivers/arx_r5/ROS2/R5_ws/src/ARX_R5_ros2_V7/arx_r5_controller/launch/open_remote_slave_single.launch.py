from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    arm_name = LaunchConfiguration('arm_name')
    arm_can_id = LaunchConfiguration('arm_can_id')
    publish_slave_status = LaunchConfiguration('remote_publish_slave_status')
    qos_depth = LaunchConfiguration('remote_qos_depth')

    return LaunchDescription([
        DeclareLaunchArgument('arm_name', default_value='arm_slave_l'),
        DeclareLaunchArgument('arm_can_id', default_value='slcan1'),
        DeclareLaunchArgument('remote_publish_slave_status', default_value='false'),
        DeclareLaunchArgument('remote_qos_depth', default_value='1'),
        Node(
            package='arx_r5_controller',
            executable='R5Controller',
            name=arm_name,
            output='screen',
            parameters=[{
                'arm_can_id': arm_can_id,
                'arm_control_type': 'remote_slave',
                'arm_pub_topic_name': 'arm_slave_l_status',
                'arm_sub_topic_name': 'arm_master_l_status',
                'arm_end_type': 0,
                'remote_publish_slave_status': ParameterValue(publish_slave_status, value_type=bool),
                'remote_qos_depth': ParameterValue(qos_depth, value_type=int),
            }],
        ),
    ])
