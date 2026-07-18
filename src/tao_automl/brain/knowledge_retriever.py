# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retrieval-Augmented Planning (RAP) from AutoML-Agent.

Retrieves external knowledge from NGC, arXiv, and web sources to inform
LLM recommendations. Summarizes findings into actionable tuning insights.
"""
import logging
from typing import Any, Dict, Optional

from tao_automl.brain.llm_client import LLMClient, first_json_object
from tao_automl.brain.prompts.autoresearch_prompts import (
    build_knowledge_summary_prompt,
)

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """Retrieves and summarizes external knowledge for informed recommendations.

    Inspired by AutoML-Agent's Retrieval-Augmented Planning (RAP) strategy.
    Uses web search APIs and LLM summarization to extract tuning insights.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        llm_params: Optional[Dict[str, Any]] = None,
        enable_web_search: bool = True,
    ):
        """Initialize the KnowledgeRetriever."""
        self.llm_client = llm_client or LLMClient(params=llm_params)
        self.enable_web_search = enable_web_search
        self._cache: Dict[str, str] = {}

    def retrieve_knowledge(
        self,
        network: str,
        metric_name: str,
        task_description: str = "",
    ) -> Optional[str]:
        """Retrieve and summarize external knowledge for a network/task."""
        cache_key = f"{network}:{metric_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        raw_knowledge = self._gather_raw_knowledge(network, metric_name, task_description)
        if not raw_knowledge:
            return None

        summary = self._summarize_knowledge(raw_knowledge, network, metric_name, task_description)
        if summary:
            self._cache[cache_key] = summary
        return summary

    def _gather_raw_knowledge(
        self, network: str, metric_name: str, task_description: str
    ) -> str:
        """Gather raw knowledge from multiple sources."""
        fragments = []

        builtin = self._get_builtin_knowledge(network)
        if builtin:
            fragments.append(f"[Built-in Knowledge]\n{builtin}")

        if self.enable_web_search:
            web_results = self._web_search(network, metric_name)
            if web_results:
                fragments.append(f"[Web Search Results]\n{web_results}")

        return "\n\n".join(fragments) if fragments else ""

    def _get_builtin_knowledge(self, network: str) -> str:
        """Return built-in tuning knowledge for known TAO networks."""
        knowledge_base = {
            "dino": (
                "DINO object detection typically works best with:\n"
                "- LR: 1e-4 to 5e-4 with cosine annealing\n"
                "- AdamW optimizer with weight_decay 0.05\n"
                "- Warmup for 500-1000 steps\n"
                "- num_queries: 300-900, num_select usually equals num_queries\n"
                "- Backbone: ResNet-50 or Swin-T for speed, Swin-L for accuracy\n"
                "- Batch size 2-4 per GPU typical for high-res images\n"
                "- Augmentation: multi-scale training helps significantly"
            ),
            "segformer": (
                "SegFormer semantic segmentation tips:\n"
                "- LR: 2e-5 to 6e-4 with polynomial LR schedule\n"
                "- AdamW optimizer, weight_decay 0.01\n"
                "- MiT-B0 for speed, MiT-B5 for accuracy\n"
                "- Batch size 4-16 depending on image resolution\n"
                "- Augmentation: random crop, photometric distortion, random flip"
            ),
            "classification_pyt": (
                "TAO classification (PyTorch) tips:\n"
                "- LR: 1e-3 to 1e-1 for training from scratch, 1e-4 to 1e-2 for fine-tuning\n"
                "- SGD with momentum 0.9 or AdamW\n"
                "- Cosine annealing schedule common\n"
                "- Backbone choice (ResNet, EfficientNet) often matters more than LR\n"
                "- Augmentation: RandAugment, Mixup, CutMix for ImageNet-scale\n"
                "- Weight decay: 1e-4 to 5e-4"
            ),
            "grounding_dino": (
                "Grounding DINO tips:\n"
                "- Similar to DINO but with text backbone considerations\n"
                "- LR: 1e-4 for visual backbone, 1e-5 for text backbone\n"
                "- Smaller batch sizes due to text encoding overhead\n"
                "- num_queries and num_select interact with grounding performance"
            ),
            "rtdetr": (
                "RT-DETR real-time detection tips:\n"
                "- LR: 1e-4 with cosine decay\n"
                "- Focus on efficiency: smaller backbone variants preferred\n"
                "- num_queries: 100-300 for speed/accuracy tradeoff\n"
                "- Batch size impacts training stability"
            ),
            "deformable_detr": (
                "Deformable DETR tips:\n"
                "- LR: 2e-4 for transformer, 2e-5 for backbone\n"
                "- num_queries: 100-300\n"
                "- Multi-scale features important\n"
                "- Training typically needs more epochs than Faster R-CNN"
            ),
        }
        return knowledge_base.get(network, "")

    def _web_search(self, network: str, metric_name: str) -> str:
        """Placeholder for web search integration."""
        # TODO: Integrate with actual search API when available
        logger.debug(
            "Web search not yet implemented. Using built-in knowledge only for %s",
            network,
        )
        return ""

    def _summarize_knowledge(
        self, raw_knowledge: str, network: str, metric_name: str, task_description: str
    ) -> Optional[str]:
        """Use LLM to summarize raw knowledge into actionable insights."""
        if not raw_knowledge.strip():
            return None

        messages = build_knowledge_summary_prompt(
            raw_knowledge=raw_knowledge,
            network=network,
            metric_name=metric_name,
            task_description=task_description or f"Training {network} model",
        )

        response = self.llm_client.chat(messages, json_mode=True, temperature=0.2)

        if not response.ok or response.json_content is None:
            return raw_knowledge

        data = first_json_object(response.json_content)
        if data is None:
            return raw_knowledge
        insights = data.get("insights", [])
        if insights:
            return "\n".join(f"- {insight}" for insight in insights)

        return raw_knowledge
