import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

import numpy as np

from vl_mapper_interface.srv import TokenizeLabels

try:
    import clip
except Exception as e:
    raise RuntimeError(
        "Failed to import clip. Install Ultralytics CLIP or OpenCLIP."
    ) from e


class TokenizerService(Node):
    def __init__(self):
        super().__init__('tokenizer_service')
        self.get_logger().info('Starting Tokenizer Service...')

        self._service_name = self.declare_parameter('tokenizer_service', '~/tokenize').get_parameter_value().string_value
        self._cb_group = ReentrantCallbackGroup()
        self._srv = self.create_service(
            TokenizeLabels,
            self._service_name,
            self.handle_tokenize,
            callback_group=self._cb_group,
        )
        self.get_logger().info(f'Tokenizer Service ready at {self._service_name}')
        
    

    def handle_tokenize(self, request: TokenizeLabels.Request, response: TokenizeLabels.Response):
        self.get_logger().info(f"Incoming tokenize request for labels: {request.labels}")
        try:
            labels = list(request.labels.labels)
            # Use CLIP's reference tokenizer to ensure exact parity
            tokens = clip.tokenize(labels)
            # tokens: torch.LongTensor [N, 77]
            ids = tokens.cpu().numpy().astype(np.int64)
            response.success = True
            response.token_ids = ids.reshape(-1).tolist()
            response.token_count = int(ids.shape[0])
        except Exception as e:
            response.success = False
            response.message = str(e)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TokenizerService()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
