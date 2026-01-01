#!/usr/bin/env python3
import sys
import threading
import tkinter as tk
from tkinter import scrolledtext

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class InstructionPublisher(Node):
    def __init__(self):
        super().__init__('gui_instruction_node')
        self._publisher_name = self.declare_parameter('instruction_topic', '/instruction').get_parameter_value().string_value
        # optional terminal mode (skip GUI)
        self._terminal_mode = self.declare_parameter('terminal_mode', False).get_parameter_value().bool_value
        self.publisher_ = self.create_publisher(String, self._publisher_name, 10)
        self._last_published = ''
        self.get_logger().info(f'GUI Node started. Publishing to "{self._publisher_name}". Waiting for input...')

    def publish_instruction(self, text):
        msg = String()
        msg.data = text
        self.publisher_.publish(msg)
        self._last_published = text
        self.get_logger().info(f'Instruction published: "{text}"')


class InstructionApp:
    def __init__(self, root: tk.Tk, ros_node: InstructionPublisher):
        self.root = root
        self.ros_node = ros_node
        self.root.title("AI Instruction Interface")
        self.root.geometry("400x350")

        # Label
        self.label = tk.Label(root, text="Enter your VLN Instruction:", font=("Arial", 12))
        self.label.pack(pady=10)

        # Text field
        self.text_area = scrolledtext.ScrolledText(root, width=40, height=10, font=("Arial", 10))
        self.text_area.pack(pady=5, padx=10)

        # Send Button
        # Buttons frame (Send + Clear)
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=15)

        self.send_btn = tk.Button(self.btn_frame, text="Send", command=self.on_send, 
                      bg="#4CAF50", fg="white", font=("Arial", 11, "bold"))
        self.send_btn.pack(side=tk.LEFT, padx=6)

        self.clear_btn = tk.Button(self.btn_frame, text="Clear", command=self.on_clear, 
                       bg="#f0f0f0", fg="black", font=("Arial", 10))
        self.clear_btn.pack(side=tk.LEFT, padx=6)

        # Status Label
        self.status_label = tk.Label(root, text="", fg="gray")
        self.status_label.pack(pady=5)

        # Last published display
        self.last_label_title = tk.Label(root, text="Last published:", font=("Arial", 10, "bold"))
        self.last_label_title.pack(pady=(8,0))
        self.last_published_var = tk.StringVar(value="<none>")
        self.last_label = tk.Label(root, textvariable=self.last_published_var, fg="#0b5394", wraplength=360, justify=tk.LEFT)
        self.last_label.pack(padx=10)

    def on_send(self):
        instruction_text = self.text_area.get("1.0", tk.END).strip()

        if instruction_text:
            self.ros_node.publish_instruction(instruction_text)
            self.status_label.config(text="Instruction published!", fg="green")
            # update last published and clear input
            self.last_published_var.set(instruction_text)
            self.text_area.delete("1.0", tk.END)
        else:
            self.status_label.config(text="Please enter text.", fg="red")

    def on_clear(self):
        self.text_area.delete("1.0", tk.END)
        self.status_label.config(text="Cleared.", fg="gray")


def main(args=None):
    rclpy.init(args=args)
    
    ros_node = InstructionPublisher()
    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    spin_thread.start()

    root = tk.Tk()
    app = InstructionApp(root, ros_node)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            ros_node.destroy_node()
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)

if __name__ == '__main__':
    main()