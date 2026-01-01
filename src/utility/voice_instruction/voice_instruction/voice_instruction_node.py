import sys
sys.modules['coverage'] = None
import threading
import time
import queue
import collections

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import numpy as np
import pyaudio
import whisper
import torch
import openwakeword
from openwakeword.model import Model

class VoiceInstructionNode(Node):
    """
    A generic ROS2 Node integrating a lightweight Wake Word Engine (openWakeWord)
    with a heavy Speech-to-Text model (Whisper).

    Workflow:
    1. Audio Loop listens continuously.
    2. openWakeWord analyzes chunks for the trigger phrase (e.g., "hey jarvis").
    3. Upon trigger, audio is captured until silence is detected (VAD).
    4. Captured audio is passed to Whisper for transcription.
    5. Resulting text is published to /instruction.
    """

    def __init__(self):
        super().__init__('voice_instruction_node')

        # --- Parameters ---
        model_size = self.declare_parameter('model_size', 'base').get_parameter_value().string_value
        ww_model_name = self.declare_parameter('wake_word_model', 'hey_jarvis_v0.1').get_parameter_value().string_value
        self.device_index = self.declare_parameter('device_index', -1).get_parameter_value().integer_value
        self.vad_threshold = self.declare_parameter('vad_threshold', 60).get_parameter_value().integer_value
        self.silence_limit = self.declare_parameter('silence_limit', 1.2).get_parameter_value().double_value
        self.pre_roll_seconds = self.declare_parameter('pre_roll_seconds', 1.0).get_parameter_value().double_value
        self.language = self.declare_parameter('language', 'en').get_parameter_value().string_value
        self.debug_mode = self.declare_parameter('debug_mode', False).get_parameter_value().bool_value
        self.instruction_topic = self.declare_parameter('instruction_topic', '/instruction').get_parameter_value().string_value

        # --- Publishers ---
        self.publisher_ = self.create_publisher(String, self.instruction_topic, 10)

        # --- AI Models Initialization ---
        self._init_models(model_size, ww_model_name)

        # --- Audio State ---
        self.audio_queue = queue.Queue()
        self.running = True
        
        # We use a deque as a circular buffer to store "pre-roll" audio
        # This ensures we don't cut off the first word of the command
        # Chunk size 1280 @ 16kHz is 80ms. 
        # Size = seconds / 0.08
        maxlen = int(self.pre_roll_seconds / 0.08)
        self.pre_roll_buffer = collections.deque(maxlen=maxlen)

        # --- Threads ---
        self.record_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.process_thread = threading.Thread(target=self._inference_loop, daemon=True)
        
        self.record_thread.start()
        self.process_thread.start()

    def _init_models(self, whisper_size, ww_name):
        """
        Helper to load both Neural Networks safely.
        """
        # 1. Load Wake Word Model
        self.get_logger().info(f"Loading Wake Word Model: {ww_name}...")
        try:
            # Automagically downloads model if not present
            openwakeword.utils.download_models() 
            self.ww_model = Model(wakeword_models=[ww_name], inference_framework="tflite")
            self.target_ww_key = ww_name
        except Exception as e:
            self.get_logger().error(f"Failed to init openWakeWord: {e}")
            raise e

        # 2. Load Whisper
        self.get_logger().info(f"Loading Whisper Model: {whisper_size}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.asr_model = whisper.load_model(whisper_size, device=device)
            self.get_logger().info(f"Models loaded. Listening for Wake Word...")
        except Exception as e:
            self.get_logger().error(f"Failed to load Whisper: {e}")
            raise e

    def _audio_loop(self):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1280 

        p = pyaudio.PyAudio()

        try:
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                            input=True, frames_per_buffer=CHUNK,
                            input_device_index=self.device_index)
        except Exception as e:
            self.get_logger().error(f"Mic Error: {e}")
            return

        is_recording_command = False
        command_frames = []
        silence_start = None

        self.get_logger().info(f"Mic listening. Threshold: {self.vad_threshold}")

        while self.running and rclpy.ok():
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_int16 = np.frombuffer(data, dtype=np.int16)
                
                # Calculate energy (volume)
                energy = np.sqrt(np.mean(audio_int16.astype(np.float32)**2))

                # DEBUG OUTPUT
                if self.debug_mode and not is_recording_command:
                    if energy > self.vad_threshold:
                        print(f"\rEnergy: {int(energy)} (ACTIVE)   ", end='', flush=True)
                    else:
                        print(f"\rEnergy: {int(energy)} (Silence)  ", end='', flush=True)

                if not is_recording_command:
                    # --- WATCHING ---
                    self.pre_roll_buffer.append(data)
                    prediction = self.ww_model.predict(audio_int16)
                    score = prediction.get(self.target_ww_key, 0.0)
                    
                    if score > 0.5: 
                        self.get_logger().info(f"\n--- WAKE WORD DETECTED ({score:.2f}) ---")
                        is_recording_command = True
                        command_frames.extend(self.pre_roll_buffer)
                        self.pre_roll_buffer.clear()
                        silence_start = None
                else:
                    # --- RECORDING ---
                    command_frames.append(data)
                    
                    if energy < self.vad_threshold:
                        if silence_start is None:
                            silence_start = time.time()
                        
                        duration_silent = time.time() - silence_start
                        if self.debug_mode:
                            print(f"\rRecording... Silence: {duration_silent:.1f}s / {self.silence_limit}s", end='', flush=True)

                        if duration_silent > self.silence_limit:
                            self.get_logger().info("\nCommand finished. Processing...")
                            full_audio = b''.join(command_frames)
                            self.audio_queue.put(full_audio)
                            command_frames = []
                            is_recording_command = False
                            self.ww_model.reset()
                    else:
                        # User is speaking -> Reset silence timer
                        if silence_start is not None and self.debug_mode:
                            print(f"\rRecording... Voice detected! Timer reset.", end='', flush=True)
                        silence_start = None

            except Exception as e:
                self.get_logger().error(f"Audio loop error: {e}")

        stream.stop_stream()
        stream.close()
        p.terminate()

    def _inference_loop(self):
        while self.running and rclpy.ok():
            try:
                audio_bytes = self.audio_queue.get(timeout=1.0)
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                result = self.asr_model.transcribe(
                    audio_np, 
                    fp16=torch.cuda.is_available(), 
                    language=self.language 
                )
                text = result['text'].strip().lstrip('Hey Jarvis,. ').strip(',:;.!?')

                if text:
                    self.get_logger().info(f"Instruction: '{text}'")
                    msg = String()
                    msg.data = text
                    self.publisher_.publish(msg)
                else:
                    self.get_logger().warn("Transcribed empty text.")
                
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Inference error: {e}")

    def destroy_node(self):
        self.running = False
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VoiceInstructionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()