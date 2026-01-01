import rclpy
import cv2
import base64
import numpy as np
import json
import threading
import os

from toon import encode 
from pprint import pprint
from pathlib import Path
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
from std_msgs.msg import String

from vl_mapper_interface.msg import Labels
from openai_base.openai_base_node import OpenAIBaseNode


class OpenAIVisionNode(OpenAIBaseNode):
    def __init__(self):
        super().__init__('openai_vision_node', 'openai_vision')
        self._cb_group = ReentrantCallbackGroup()
        
        if not self.init_ok:
            return

        self.setup_ros_interfaces()
        self.read_system_prompt()
        self.read_prompt_template()
        self.get_logger().info("OpenAI Vision Node ready.")


    def setup_parameters(self):
        super().setup_parameters()

        # Topics
        self._image_topic = self.declare_parameter(
            'image_topic', '/camera/color/image_raw'
        ).get_parameter_value().string_value
        self._instruction_topic = self.declare_parameter(
            'instruction_topic', '/task/instruction'
        ).get_parameter_value().string_value
        self._labels_topic = self.declare_parameter(
            'labels_topic', '/vl_mapper/labels'
        ).get_parameter_value().string_value


    def setup_ros_interfaces(self):
        self._latest_image = None
        self._image_lock = threading.Lock()

        self._image_sub = self.create_subscription(
            Image,
            self._image_topic,
            self.image_callback,
            10,
            callback_group=self._cb_group
        )
        self._instruction_sub = self.create_subscription(
            String,
            self._instruction_topic,
            self.instruction_callback,
            10,
            callback_group=self._cb_group
        )
        self._labels_pub = self.create_publisher(
            Labels,
            self._labels_topic,
            10
        )


    def read_system_prompt(self) -> None:
        default_system_prompt = (
            "You are a vision assistant for a robot. ",
            "Your task is to analyze the image based on the user instruction and return a JSON array of 3 to 5 strings.\n",
            "Rules:\n"
            "The FIRST string must be the target object mentioned or implied in the instruction (e.g., if instruction is 'go to the fridge', first label is 'fridge').\n",
            "If the target is not visible, or for the remaining strings, list prominent objects visible in the image that serve as landmarks.\n",
            "Use specific, short phrases (1-3 words), e.g., 'fire extinguisher' instead of just 'extinguisher'.\n",
            "Return ONLY the raw JSON array. Do not use Markdown formatting (no ```json blocks)."
        )
        self._system_prompt = self.read_text_file_or_default(
            self._system_prompt_file, default_system_prompt, "system_prompt",
            write_path=self._default_system_prompt_file
        )


    def read_prompt_template(self) -> None:
        default_prompt_template = 'instruction: {instruction}'
        self._prompt_template = self.read_text_file_or_default(
            self._prompt_template_file, default_prompt_template, "prompt_template",
            write_path=self._default_prompt_template_file
        )
        if '{instruction}' not in self._prompt_template:
            self.get_logger().warn("Prompt template missing '{instruction}' placeholder. Appending it.")
            self._prompt_template = f'instruction: "{{instruction}}".\n{self._prompt_template}'


    def image_callback(self, msg: Image):
        with self._image_lock:
            self._latest_image = msg


    def instruction_callback(self, msg: String):
        instruction = msg.data.strip()
        if not instruction:
            self.get_logger().warn("Empty instruction, skipping.")
            return

        with self._image_lock:
            image_msg = self._latest_image

        if image_msg is None:
            self.get_logger().warn("No image available yet, skipping.")
            return

        # Decode ROS Image to NumPy (BGR for OpenCV)
        try:
            cv_bgr = self.decode_ros_image_to_bgr(image_msg)
        except Exception as e:
            self.get_logger().error(f"Image decode failed: {e}")
            return

        try:
            base64_image = self.encode_image_to_base64(cv_bgr)
        except Exception as e:
            self.get_logger().error(f"Image base64 encode failed: {e}")
            return

        try:
            prompt_text = self._prompt_template.format(instruction=instruction)
        except Exception as e:
            self.get_logger().error(f"Prompt format failed: {e}")
            return

        try:
            messages = [
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_text},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"},
                    ],
                },
            ]

            resp = self._client.responses.create(
                model=self._model,
                input=messages,
            )
            
            self.publish_labels_from_response(resp.output_text)
        except Exception as e:
            self.get_logger().error(f"OpenAI request failed: {e}")


    def decode_ros_image_to_bgr(self, msg: Image) -> np.ndarray:
        # Reshape raw buffer
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        enc = (msg.encoding or '').lower()

        if enc in ('bgr8', 'rgb8'):
            cv_img = buf.reshape(msg.height, msg.width, 3)
            if enc == 'rgb8':
                cv_img = cv_img[:, :, ::-1]  # RGB -> BGR
            return cv_img

        if enc in ('mono8', '8uc1'):
            gray = buf.reshape(msg.height, msg.width)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Fallback: attempt 3-channel reshape
        try:
            cv_img = buf.reshape(msg.height, msg.width, 3)
            return cv_img
        except Exception:
            raise ValueError(f"Unsupported image encoding: '{msg.encoding}'")


    def encode_image_to_base64(self, cv_bgr: np.ndarray) -> str:
        success, buffer = cv2.imencode('.jpg', cv_bgr)
        if not success:
            raise RuntimeError("cv2.imencode failed for JPEG.")
        return base64.b64encode(buffer).decode('utf-8')


    def publish_labels_from_response(self, response_content: str):
        # Extract JSON array from content, tolerate possible markdown fences
        try:
            raw = response_content.strip()
            if "```" in raw:
                parts = raw.split("```")
                if len(parts) >= 2:
                    raw = parts[1]
                raw = raw.replace('json\n', '').replace('JSON\n', '').strip()

            labels_list = json.loads(raw)
            if not isinstance(labels_list, list):
                self.get_logger().warn("Response is not a JSON array, ignoring.")
                return

            labels_clean = [str(x).strip() for x in labels_list if str(x).strip()]
            if not labels_clean:
                self.get_logger().warn("No valid labels parsed from response.")
                return

            out = Labels()
            out.labels = labels_clean
            self._labels_pub.publish(out)
            self.get_logger().info(f"Published labels: {labels_clean}")
        except Exception as e:
            self.get_logger().error(f"Failed to parse OpenAI response: {e}")
            self.get_logger().error(f"Content: {response_content}")


def main(args=None):
    rclpy.init(args=args)
    node = OpenAIVisionNode()
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