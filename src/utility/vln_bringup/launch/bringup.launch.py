#!/usr/bin/env python3
import os
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, ExecuteProcess, LogInfo 
from launch.substitutions import LaunchConfiguration, TextSubstitution, PythonExpression
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()
    
    # --- Arguments ---
    # topics
    ld.add_action(DeclareLaunchArgument(
        'rgb_topic', default_value='/scout2/camera/color/image_raw',
        description='RGB image topic for YOLO'))
    ld.add_action(DeclareLaunchArgument(
        'depth_topic', default_value='/scout2/camera/depth/image_rect_raw',
        description='Depth image topic for YOLO'))
    ld.add_action(DeclareLaunchArgument(
        'camera_info_topic', default_value='/scout2/camera/color/camera_info',
        description='Camera info topic for YOLO'))
    ld.add_action(DeclareLaunchArgument(
        'odom_topic', default_value='/odom',
        description='Odometry topic to sync with images'))
    ld.add_action(DeclareLaunchArgument(
        'labels_topic', default_value='/labels',
        description='Labels topic for YOLO'))
    ld.add_action(DeclareLaunchArgument(
        'instruction_topic', default_value='/instruction',
        description='The instruction topic for label generation'))
    ld.add_action(DeclareLaunchArgument(
        'detections_topic', default_value='/vl_mapper/detections_3d',
        description='3D Detections topic from VL Mapper'))
    
    # ros
    ld.add_action(DeclareLaunchArgument(
        'yolo_service', default_value='/yolo_service/detect_labels',
        description='Service name of YOLO detect service (relative to node ns)'))
    ld.add_action(DeclareLaunchArgument(
        'tokenizer_service', default_value='/tokenizer_service/tokenize',
        description='Tokenizer service full name'))
    ld.add_action(DeclareLaunchArgument(
        'use_onnx', default_value='true',
        description='Use ONNX models (true/false)'))
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation (Gazebo) clock if true'))
    ld.add_action(DeclareLaunchArgument(
        'instruction_node', default_value='none',
        description='The instruction publisher node to use (none/gui/voice)'))
    
    # models
    ld.add_action(DeclareLaunchArgument(
        'clip_model', default_value=os.path.join(
            os.path.expanduser('~'), 'yolo_world_models', 'clip_text_encoder_slim.onnx'),
        description='Path to CLIP text encoder ONNX'))
    ld.add_action(DeclareLaunchArgument(
        'vision_model', default_value=os.path.join(
            os.path.expanduser('~'), 'yolo_world_models', 'yoloworld_vision_slim.onnx'),
        description='Path to YOLOWorld vision ONNX'))
    ld.add_action(DeclareLaunchArgument(
        'ultralytics_model', default_value=os.path.join(
            os.path.expanduser('~'), 'yolo_world_models', 'yolov8m-worldv2.pt'),
        description='Path to YOLO Ultralytics ONNX'))
    ld.add_action(DeclareLaunchArgument(
        'openai_model', default_value='gpt-5-mini',
        description='Name of the OpenAI model (e.g.: "gpt-5-mini", "gpt-4o",...).'))
    
    # others
    ld.add_action(DeclareLaunchArgument(
        'stamp_tolerance_ms', default_value='200',
        description='Allowed timestamp tolerance (ms) when matching RGB frames'))
    ld.add_action(DeclareLaunchArgument(
        'cache_size', default_value='50',
        description='RGB LRU cache size'))
    ld.add_action(DeclareLaunchArgument(
        'log_level', default_value='info',
        description='Logging level for all nodes (debug, info, warn, error, fatal)'))


    # --- Configurations ---
    # topics
    rgb_topic = LaunchConfiguration('rgb_topic')
    depth_topic = LaunchConfiguration('depth_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')
    odom_topic = LaunchConfiguration('odom_topic')
    labels_topic = LaunchConfiguration('labels_topic')
    instruction_topic = LaunchConfiguration('instruction_topic')
    detections_topic = LaunchConfiguration('detections_topic')

    # ros
    yolo_service = LaunchConfiguration('yolo_service')
    tokenizer_service = LaunchConfiguration('tokenizer_service')
    log_level = LaunchConfiguration('log_level')
    use_onnx = LaunchConfiguration('use_onnx')
    use_sim_time = LaunchConfiguration('use_sim_time')
    instruction_node = LaunchConfiguration('instruction_node')

    # others
    stamp_tol = LaunchConfiguration('stamp_tolerance_ms')
    cache_size = LaunchConfiguration('cache_size')
    openai_model = LaunchConfiguration('openai_model')


    # --- Nodes ---
    ld.add_action(Node(
        package='tokenizer',
        executable='tokenizer_node',
        name='tokenizer_service',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
        parameters=[{
            'tokenizer_service': tokenizer_service,
            'use_sim_time': use_sim_time,
        }],
        condition=IfCondition(use_onnx),
    ))

    ld.add_action(Node(
        package='yolo_onnx',
        executable='yolo_onnx_node',
        name='yolo_onnx_service',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
        parameters=[{
            'rgb_topic': rgb_topic,
            'yolo_service': yolo_service,
            'tokenizer_service': tokenizer_service,
            'clip_model': LaunchConfiguration('clip_model'),
            'vision_model': LaunchConfiguration('vision_model'),
            'stamp_tolerance_ms': stamp_tol,
            'cache_size': cache_size,
            'use_sim_time': use_sim_time,
        }],
        condition=IfCondition(use_onnx),
    ))
    
    ld.add_action(Node(
        package='yolo_world',
        executable='yolo_world_node',
        name='yolo_ultralytics_service',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
        parameters=[{
            'model_path': LaunchConfiguration('ultralytics_model'),
            'rgb_topic': rgb_topic,
            'yolo_service': yolo_service,
            'stamp_tolerance_ms': stamp_tol,
            'cache_size': cache_size,
            'use_sim_time': use_sim_time,
        }],
        condition=UnlessCondition(use_onnx),
    ))

    ld.add_action(Node(
        package='openai_vision',
        executable='openai_vision_node',
        name='openai_vision',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
        parameters=[{
            'image_topic': rgb_topic,
            'openai_model': openai_model,
            'instruction_topic': instruction_topic,
            'labels_topic': labels_topic,
            'use_sim_time': use_sim_time,
        }],
    ))

    ld.add_action(Node(
        package='openai_detections',
        executable='openai_detections_node',
        name='openai_detections',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
        parameters=[{
            'openai_model': openai_model,
            'instruction_topic': instruction_topic,
            'detections_topic': detections_topic,
            # 'final_detection_topic': final_detection_topic,
            'use_sim_time': use_sim_time,
        }],
    ))

    ld.add_action(Node(
        package='vl_mapper',
        executable='vl_mapper_node',
        name='vl_mapper_node',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
        parameters=[{
            'rgb_topic': rgb_topic,
            'depth_topic': depth_topic,
            'camera_info_topic': camera_info_topic,
            'labels_topic': labels_topic,
            'yolo_service': yolo_service,
            'odom_topic': odom_topic,
            'use_sim_time': use_sim_time,
            'detections_topic': detections_topic,
        }],
    ))

    ld.add_action(Node(
        package='gui_instruction',
        executable='gui_instruction_node',
        name='gui_instruction_node',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
        parameters=[{
            'instruction_topic': instruction_topic,
            'use_sim_time': use_sim_time,
        }],
        condition=IfCondition(PythonExpression(["'", instruction_node, "' == 'gui'"])),
    ))

    ld.add_action(Node(
        package='voice_instruction',
        executable='voice_instruction_node',
        name='voice_instruction_node',
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
        parameters=[{
            'instruction_topic': instruction_topic,
            'use_sim_time': use_sim_time,
        }],
        condition=IfCondition(PythonExpression(["'", instruction_node, "' == 'voice'"])),
    ))


    # --- TEST STUFF ---
    ld.add_action(DeclareLaunchArgument(
        'labels', default_value='["chair", "table", "person"]',
        description='List of labels for YOLO detection'))
    
    labels = LaunchConfiguration('labels')

    publish_command = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--once', labels_topic, 'vl_mapper_interface/msg/Labels',
            [TextSubstitution(text='{"labels": '), labels, TextSubstitution(text='}')]
        ],
        shell=False
    )

    ld.add_action(TimerAction(
        period=10.0,
        actions=[
            LogInfo(msg="Publishing Labels..."),
            publish_command
        ],
        condition=IfCondition(PythonExpression(["'", instruction_node, "' == 'none'"])),
     ))

    return ld
