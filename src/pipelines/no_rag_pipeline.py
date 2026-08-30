from __future__ import annotations

from src.llm.llm_factory import create_chat_llm
from src.pipelines.base_pipeline import BaseRAGPipeline


class NoRAGPipeline(BaseRAGPipeline):
    pipeline_name = "no_rag"

    def __init__(self) -> None:
        super().__init__()
        self.llm = None

    def run(self, question: str) -> dict:
        self.llm = create_chat_llm(self.settings)
        messages = [
            ("system", self.prompts.get("system_prompt", "")),
            ("human", question),
        ]
        answer = self.llm.invoke(messages).content

        payload = {
            "pipeline": self.pipeline_name,
            "question": question,
            "answer": answer,
            "retrieved": [],
        }
        output_file = self.save_run(payload)
        self.logger.info("Run saved to %s", output_file)
        return payload
