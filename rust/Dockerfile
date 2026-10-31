# Borg runtime image: the native binary only. CUDA/TensorRT and the admitted
# assets are read-only host mounts supplied by the existing NVIDIA runtime.
# Build with the governed ARM release copied into the context as chatterbox-rs.
FROM scratch
COPY chatterbox-rs /usr/local/bin/myth
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64
ENTRYPOINT ["/usr/local/bin/myth"]
