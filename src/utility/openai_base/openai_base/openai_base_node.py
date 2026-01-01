import os

from pathlib import Path
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from openai import OpenAI

class OpenAIBaseNode(Node):
    def __init__(self, node_name: str, package_name: str):
        super().__init__(node_name)
        self.init_ok = True
        self._client = None

        # Parameters
        self._model = None
        self._system_prompt_file = None
        self._prompt_template_file = None

        try:
            prompts_share = Path(get_package_share_directory(package_name)) / 'prompts'
            self._default_system_prompt_file = str(prompts_share / 'default_system.txt')
            self._default_prompt_template_file = str(prompts_share / 'default_prompt.txt')
        except Exception as e:
            self.get_logger().error(f"Failed to find share directory for {package_name}: {e}")
            self._default_system_prompt_file = ''
            self._default_prompt_template_file = ''

        self.setup_parameters()
        if not self.setup_openai_client():
            self.init_ok = False
            return


    def setup_parameters(self):
        self._model = self.declare_parameter(
            'openai_model', 'gpt-5-mini'
        ).get_parameter_value().string_value

        self._system_prompt_file = self.declare_parameter(
            'system_prompt_file', self._default_system_prompt_file
        ).get_parameter_value().string_value

        self._prompt_template_file = self.declare_parameter(
            'prompt_template_file', self._default_prompt_template_file
        ).get_parameter_value().string_value


    def setup_openai_client(self) -> bool:
        try:
            api_key = os.environ.get('OPENAI_API_KEY', '')
            if not api_key:
                self.get_logger().fatal("Environment variable OPENAI_API_KEY is not set.")
                return False

            self._client = OpenAI(api_key=api_key)
            return True
        except Exception as e:
            self.get_logger().fatal(f"OpenAI client init failed: {e}")
            return False


    def read_text_file_or_default(self, path: str, default_text: str, label: str, write_path: str = None) -> str:
        try:
            expanded = os.path.expandvars(os.path.expanduser(path or ''))
            file_exists = expanded and os.path.isfile(expanded)
            content = ""

            if file_exists:
                with open(expanded, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

            if content:
                self.get_logger().info(f"Loaded {label} from: {expanded}")
                return content

            # File is missing or empty
            target_path = write_path if write_path else expanded
            if target_path:
                target_expanded = os.path.expandvars(os.path.expanduser(target_path))

                if file_exists:
                    self.get_logger().warn(f"{label} file is empty: {expanded}. Overwriting default to {target_expanded}.")
                else:
                    self.get_logger().warn(f"{label} file not found: {expanded}. Creating default at {target_expanded}.")

                directory = os.path.dirname(target_expanded)
                if directory:
                    os.makedirs(directory, exist_ok=True)
                with open(target_expanded, 'w', encoding='utf-8') as f:
                    f.write(default_text)
                self.get_logger().info(f"Created default {label} file at: {target_expanded}")

        except Exception as e:
            self.get_logger().warn(f"Failed to read/write {label} file '{path}': {e}. Using default.")
        return default_text