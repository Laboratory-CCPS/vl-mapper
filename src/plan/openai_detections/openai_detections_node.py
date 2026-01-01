import rclpy
import os
import time
import threading
import re

from toon import encode 
from pathlib import Path
from typing import List, Optional, Tuple

from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from vision_msgs.msg import Detection3DArray, Detection3D
from std_msgs.msg import String
from rosidl_runtime_py import message_to_ordereddict
from openai_base.openai_base_node import OpenAIBaseNode


class OpenAIDetectionsNode(OpenAIBaseNode):
    def __init__(self):
        super().__init__('openai_detections_node', 'openai_detections')
        self._cb_group = ReentrantCallbackGroup()
        
        if not self.init_ok:
            return

        self.read_system_prompt()
        self.read_prompt_template()
        self.setup_ros_interfaces()

        self._last_request_time = 0.0
        self.get_logger().info("OpenAI Detections Node ready.")

    def setup_parameters(self):
        super().setup_parameters()

        self._temperature = self.declare_parameter('temperature', 0.0).get_parameter_value().double_value
        self._min_request_period = self.declare_parameter('min_request_period', 1.5).get_parameter_value().double_value

        self._detections_topic = self.declare_parameter('detections_topic', '/detections').get_parameter_value().string_value
        self._instruction_topic = self.declare_parameter('instruction_topic', '/task/query').get_parameter_value().string_value
        self._final_detection_topic = self.declare_parameter('final_detection_topic', '/vl_mapper/final_detection').get_parameter_value().string_value

    def read_system_prompt(self):
        default = (
            "You select exactly one detection result id matching a natural language instruction. "
            "You will receive a list of detections with result ids, scores, centers and sizes plus an instruction "
            "like 'Find the table next to the two chairs'. Return ONLY one existing result id string from the list. "
            "If none fits, return the highest-confidence id. No quotes, no extra words."
        )
        self._system_prompt = self.read_text_file_or_default(self._system_prompt_file, default, "system_prompt")

    def read_prompt_template(self):
        default = (
            "Instruction: {instruction}\n"
            "Detections ({count} total):\n"
            "{detection_lines}\n"
            "Choose ONE result id that best matches the instruction (or highest confidence if none). "
            "Return ONLY that id."
        )
        self._prompt_template = self.read_text_file_or_default(self._prompt_template_file, default, "selection_prompt")
        for ph in ("{instruction}", "{detection_lines}", "{count}"):
            if ph not in self._prompt_template:
                self._prompt_template += f"\n{ph}"

    def setup_ros_interfaces(self):
        self._det_sub = self.create_subscription(
            Detection3DArray,
            self._detections_topic,
            self.detections_callback,
            10,
            callback_group=self._cb_group,
        )
        self._instruction_sub = self.create_subscription(
            String,
            self._instruction_topic,
            self.instruction_callback,
            10,
            callback_group=self._cb_group,
        )
        self._final_pub = self.create_publisher(
            Detection3D,
            self._final_detection_topic,
            10,
        )
        self._latest_detections: Optional[Detection3DArray] = None
        self._det_lock = threading.Lock()

    def detections_callback(self, msg: Detection3DArray):
        with self._det_lock:
            self._latest_detections = msg

    def instruction_callback(self, msg: String):
        instruction = msg.data.strip()
        if not instruction:
            self.get_logger().warn("Empty instruction.")
            return

        with self._det_lock:
            det_msg = self._latest_detections

        if det_msg is None or not det_msg.detections:
            self.get_logger().warn("No detections available.")
            return

        now = time.time()
        if (now - self._last_request_time) < self._min_request_period:
            self.get_logger().warn("Throttled.")
            return
        self._last_request_time = now

        det_lines, result_ids = self.format_detections(det_msg)
        if not result_ids:
            self.get_logger().warn("No result ids in detections.")
            return

        try:
            prompt = self._prompt_template.format(
                instruction=instruction,
                detection_lines="\n".join(det_lines),
                count=len(det_msg.detections)
            )
        except Exception as e:
            self.get_logger().error(f"Prompt format error: {e}")
            return

        try:
            resp = self._client.responses.create(
                model=self._model,
                input=[
                    {"role": "system", "content": self._system_prompt},
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}]
                    },
                ],
                temperature=float(self._temperature),
            )
        except Exception as e:
            self.get_logger().error(f"OpenAI request failed: {e}")
            return

        output_text = getattr(resp, "output_text", "") or self._extract_output_text(resp)
        if not output_text:
            self.get_logger().warn("Empty LLM output.")
            return

        chosen_id = self.extract_chosen_id(output_text, result_ids)
        if not chosen_id:
            self.get_logger().warn(f"No valid id parsed from: {output_text}")
            return

        det = self.find_detection_by_result_id(det_msg, chosen_id)
        if det is None:
            self.get_logger().warn(f"Chosen id '{chosen_id}' not found.")
            return

        self._final_pub.publish(det)
        self.get_logger().info(f"Published detection id {det.id} for instruction: {instruction}")

    def format_detections(self, msg: Detection3DArray) -> Tuple[List[str], List[str]]:
        lines: List[str] = []
        ids: List[str] = []
        for det in msg.detections:
            det_dict = message_to_ordereddict(det)
            lines.append(encode(det_dict))

            for res in det.results:
                rid = getattr(res, "id", "") or getattr(getattr(res, "hypothesis", res), "id", "")
                if rid:
                    ids.append(str(rid))
        return lines, ids

    def _extract_output_text(self, resp) -> str:
        parts = []
        for item in getattr(resp, "output", []) or []:
            if isinstance(item, dict):
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        parts.append(c.get("text", ""))
        return " ".join(parts).strip()

    def extract_chosen_id(self, text: str, valid_ids: List[str]) -> Optional[str]:
        raw = text.strip()
        if raw in valid_ids:
            return raw
        cleaned = raw.strip('"').strip("'").strip()
        if cleaned in valid_ids:
            return cleaned
        for token in re.split(r'\s+', cleaned):
            if token in valid_ids:
                return token
        return None

    def find_detection_by_result_id(self, msg: Detection3DArray, rid: str) -> Optional[Detection3D]:
        for det in msg.detections:
            for res in det.results:
                res_id = getattr(res, "id", "") or getattr(getattr(res, "hypothesis", res), "id", "")
                if res_id == rid:
                    return det
        return None


def main(args=None):
    rclpy.init(args=args)
    node = OpenAIDetectionsNode()
    if node.init_ok and rclpy.ok():
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()