FROM vllm/vllm-openai-rocm@sha256:c6cb4ab021e31691f63a5757278153fbe728a71c722651a4549ddbb5ac3067ab

# Build from the repository root. No compiler cache, model weights, or native
# libraries from the source machine are copied into the image.
COPY vllm/ /opt/mm3/source/vllm/
COPY benchmarks/minimax_m3_nightly/ /opt/mm3/handoff/
RUN uv venv --offline --python /usr/bin/python3 --system-site-packages /tmp/mm3-build-venv \
    && /tmp/mm3-build-venv/bin/python -B /opt/mm3/handoff/install.py \
       --apply --repository /opt/mm3/source --arm n18

ENV MM3_SPARSE_PA_PARTITION_EXPERIMENT=1 \
    MM3_DRAFT_HEAD_PREPACK_EXPERIMENT=1 \
    MM3_SHARDED_GREEDY_VERIFY_EXPERIMENT=1 \
    MM3_FUSED_DRAFT_ARGMAX_EXPERIMENT=1 \
    MM3_N07_FUSED_DRAFT=1 \
    MM3_N14_G1_TN64_K256=0 \
    MM3_N16_G1_TN64_K128=1 \
    MM3_N18_BF16X3_ROUTER=1
