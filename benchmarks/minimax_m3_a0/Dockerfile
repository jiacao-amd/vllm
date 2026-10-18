ARG BASE_IMAGE=local/mm3-vllm-2a02f6e:latest
FROM ${BASE_IMAGE}
ARG ARM=a0
COPY . /opt/mm3-ablation
RUN uv venv --offline --python /usr/bin/python3 --system-site-packages /opt/mm3/.venv \
    && /opt/mm3/.venv/bin/python -B /opt/mm3-ablation/tools/apply_arm.py \
       --target /usr/local/lib/python3.12/dist-packages --arm "${ARM}"
ENV PYTHONDONTWRITEBYTECODE=1
LABEL mm3.ablation.arm="${ARM}" \
      mm3.ablation.base="sha256:2695afe0011d4a5c9ab394a8252a590fe0c0ee4497d483e50606e996762952a7"
ENTRYPOINT ["/opt/mm3/.venv/bin/python", "-B", "-m", "vllm.entrypoints.cli.main"]
