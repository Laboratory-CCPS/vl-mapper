FROM ros:humble-ros-base

# 1. System-Abhängigkeiten installieren
RUN apt-get update && apt-get install -y \
    python3-pip \
    portaudio19-dev \
    git \
    wget \
    nano \
    && rm -rf /var/lib/apt/lists/*

# 2. Workspace vorbereiten
WORKDIR /root/vln_ws

# 3. Workspace vorbereiten
COPY requirements.txt .
COPY install_onnx_cpp.sh .

# 4. ONNX & Python Dependencies installieren
RUN chmod +x install_onnx_cpp.sh && ./install_onnx_cpp.sh
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# 5. Copy Workspace Dateien
COPY src ./src
COPY scripts ./scripts

# 6. ROS-Abhängigkeiten nachinstallieren (rosdep)
RUN apt-get update && rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y

# 7. download models
RUN mkdir -p /root/vln_ws/yolo_world_models && \
    # Export
    python3 /root/vln_ws/scripts/export_yoloworld.py \
        --weights /root/vln_ws/yolo_world_models/yolov8m-worldv2.pt \
        --out-dir /root/vln_ws/yolo_world_models \
        --max-labels 3 && \
    # Löschen
    find /root/vln_ws/yolo_world_models/ -type f -not -name '*slim.onnx' -delete && \
    rm -rf /root/.cache/pip

# 7. Workspace bauen
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"

# 8. Automatisch sourcen, wenn der Container startet
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
RUN echo "source /root/vln_ws/install/setup.bash" >> /root/.bashrc

# Standard-Kommando (Shell starten)
CMD ["/bin/bash"]