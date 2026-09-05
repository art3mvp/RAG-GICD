from __future__ import annotations

from src.embeddings.embedding_factory import create_embeddings
from src.llm.llm_factory import create_chat_llm
from src.llm.prompt_builder import build_messages
from src.pipelines.base_pipeline import BaseRAGPipeline
from src.retrievers.dense_retriever import DenseRetriever
from src.stores.faiss_store import FAISSStore


class NaiveRAGPipeline(BaseRAGPipeline):
    pipeline_name = "naive"

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = None
        self.llm = None
        self.store = FAISSStore(self.settings.faiss_index_dir)

    def prepare_index(self, chunks) -> None:
        if self.store.exists():
            self.store.load(self.embeddings)
            self.logger.info("Loaded existing FAISS index")
            return
        self.store.build(chunks, self.embeddings)
        self.store.save()
        self.logger.info("Built and saved FAISS index")

    def run(self, question: str) -> dict:
        chunks = self.load_processed_chunks()
        if not chunks:
            return {"question": question, "answer": "I do not know.", "retrieved": []}

        self.embeddings = create_embeddings(self.settings)
        self.llm = create_chat_llm(self.settings)
        self.prepare_index(chunks)
        retriever = DenseRetriever(self.store, method_name="faiss")
        retrieved = retriever.retrieve(question, self.settings.top_k)
        messages = build_messages(question, retrieved, self.prompts)
        answer = self.llm.invoke(messages).content

        payload = {
            "pipeline": self.pipeline_name,
            "question": question,
            "answer": answer,
            "retrieved": [item.__dict__ for item in retrieved],
        }
        output_file = self.save_run(payload)
        self.logger.info("Run saved to %s", output_file)
        return payload
