import rclpy
import numpy as np
import threading
from collections import OrderedDict
from rclpy.node import Node
from ultralytics.models import YOLOWorld
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from vl_mapper_interface.srv import DetectAtStamp


class YoloServiceServer(Node):
    def __init__(self):
        super().__init__('yolo_service_server')
        self.get_logger().info("Starting YOLO World Service Server...")

        self.setup_parameters()

        # Load Model
        self.model = YOLOWorld(self._model_path)
        self._model_lock = threading.Lock()
        self.previous_labels = set()

        # Cache recent RGB images keyed by (sec, nanosec) with LRU eviction
        self._rgb_cache = OrderedDict()
        self._cache_lock = threading.Lock()

        self.setup_services()
        self.get_logger().info(f"YOLO Service Server is ready at {self._service_name}.")


    def setup_services(self):
        self._cb_group = ReentrantCallbackGroup()

        self._rgb_sub = self.create_subscription(
            Image,
            self._rgb_topic,
            self.rgb_callback,
            10,
            callback_group=self._cb_group,
        )

        self._srv = self.create_service(
            DetectAtStamp,
            self._service_name,
            self.handle_detect,
            callback_group=self._cb_group,
        )


    def setup_parameters(self):
        self._model_path = self.declare_parameter('model_path', '/home/josua/yolo_world_weights/yolov8l-worldv2.pt').get_parameter_value().string_value
        self._rgb_topic = self.declare_parameter('viz_rgb_topic', '/camera/color/image_raw').get_parameter_value().string_value
        self._service_name = self.declare_parameter('yolo_service', 'detect_labels').get_parameter_value().string_value
        self._cache_max = self.declare_parameter('cache_size', 20).get_parameter_value().integer_value
        self._stamp_tolerance_ms = self.declare_parameter('stamp_tolerance_ms', 0).get_parameter_value().integer_value
        self._conf = self.declare_parameter('conf_threshold', 0.35).get_parameter_value().double_value
        self._iou = self.declare_parameter('iou_threshold', 0.5).get_parameter_value().double_value


    def rgb_callback(self, msg: Image):
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        with self._cache_lock:
            # Insert or move to end (most recently used)
            self._rgb_cache[key] = msg
            self._rgb_cache.move_to_end(key)
            # Evict while over capacity
            while len(self._rgb_cache) > max(1, self._cache_max):
                self._rgb_cache.popitem(last=False)


    def handle_detect(self, request: DetectAtStamp.Request, response: DetectAtStamp.Response):
        # Get corresponding RGB frame by timestamp
        stamp = request.stamp
        key = (stamp.sec, stamp.nanosec)
        with self._cache_lock:
            rgb_msg: Image = self._rgb_cache.get(key)
            # If not found and tolerance configured, find nearest within tolerance
            if rgb_msg is None and self._stamp_tolerance_ms > 0 and len(self._rgb_cache) > 0:
                target_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
                tol_ns = int(self._stamp_tolerance_ms) * 1_000_000
                nearest_key = None
                nearest_delta = None
                for (s, ns), m in self._rgb_cache.items():
                    cur_ns = int(s) * 1_000_000_000 + int(ns)
                    delta = abs(cur_ns - target_ns)
                    if nearest_delta is None or delta < nearest_delta:
                        nearest_delta = delta
                        nearest_key = (s, ns)
                if nearest_delta is not None and nearest_delta <= tol_ns:
                    rgb_msg = self._rgb_cache[nearest_key]
        if rgb_msg is None:
            response.success = False
            response.message = f"No RGB frame with stamp {stamp.sec}.{stamp.nanosec} in cache"
            response.detections = Detection2DArray()
            return response

        labels = set(request.labels.labels)

        # Decode image to numpy RGB
        numpy_flat = np.frombuffer(rgb_msg.data, np.uint8)
        try:
            cv_image = numpy_flat.reshape(rgb_msg.height, rgb_msg.width, 3)
        except ValueError:
            response.success = False
            response.message = "RGB image data size does not match HxW x3"
            response.detections = Detection2DArray()
            return response
        if rgb_msg.encoding.lower() == 'bgr8':
            cv_image = cv_image[:, :, ::-1]

        # only update model if labels changed
        if labels != self.previous_labels:
            with self._model_lock:
                self.model.set_classes(list(labels))
            self.previous_labels = labels

        # YOLO World inference
        with self._model_lock:
            results = self.model.predict(cv_image, conf=self._conf, iou=self._iou, verbose=False)

        # create result
        detection_array = Detection2DArray()
        detection_array.header = rgb_msg.header

        if results and len(results) > 0:
            for res in results[0].boxes:
                det = Detection2D()
                det.header = rgb_msg.header

                # Bounding Box (normalized xywh in [0,1])
                box = res.xywhn[0]
                det.bbox.center.position.x = float(box[0])
                det.bbox.center.position.y = float(box[1])
                det.bbox.size_x = float(box[2])
                det.bbox.size_y = float(box[3])

                # Hypothesis (class and confidence)
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = self.model.names[int(res.cls[0])]
                hypothesis.hypothesis.score = float(res.conf[0])
                det.results.append(hypothesis)

                detection_array.detections.append(det)

        response.detections = detection_array
        response.success = True
        response.message = "ok"
        return response



def main(args=None):
    rclpy.init(args=args)
    node = YoloServiceServer()
    # Use MultiThreadedExecutor so subscription and service can run concurrently
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()
