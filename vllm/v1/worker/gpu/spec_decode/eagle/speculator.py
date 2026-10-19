# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
    TargetDependentARSpeculator,
)


class EagleSpeculator(TargetDependentARSpeculator):
    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        model = load_eagle_model(target_model, self.vllm_config)
        from ._n06_draft_head.runtime import install_loaded_draft

        self._n06_draft_head = install_loaded_draft(model, target_model, self.vllm_config)
        return model

    def capture(self) -> None:
        installation = getattr(self, "_n06_draft_head", None)
        if installation is None:
            return super().capture()
        with installation.prepare_capture():
            return super().capture()
