from typing import Dict, List

import bm25s
import numpy as np
import weave
from sentence_transformers import SentenceTransformer


class BM25Retriever(weave.Model):
    weave_dataset_address: str
    _corpus: List[Dict[str, str]] = []
    _index: bm25s.BM25 = None

    def __init__(self, weave_dataset_address: str):
        super().__init__(weave_dataset_address=weave_dataset_address)
        self._index = bm25s.BM25()
        dataset_rows = weave.ref(self.weave_dataset_address).get().rows
        self._corpus = [
            dict(row) for row in weave.ref(self.weave_dataset_address).get().rows
        ]
        corpus_tokens = bm25s.tokenize(
            [
                row["text"] + "\n\n# Image Descriptions\n" + row["image_descriptions"]
                for row in dataset_rows
            ]
        )
        self._index.index(corpus_tokens, show_progress=True)

    @weave.op()
    def search(self, query: str, top_k: int = 5):
        query_tokens = bm25s.tokenize(query)
        results, scores = self._index.retrieve(
            query_tokens, corpus=self._corpus, k=top_k, show_progress=True
        )
        output = []
        for idx in range(results.shape[1]):
            output.append(
                {
                    "text": results[0, idx]["text"],
                    "image_descriptions": results[0, idx]["image_descriptions"],
                    "source": results[0, idx]["pdf_file"],
                    "score": scores[0, idx],
                    "dataset_idx": idx,
                }
            )
        return output

    @weave.op()
    def predict(self, query: str, top_k: int = 5):
        return self.search(query, top_k)


class BGERetriever(weave.Model):
    model_name: str
    weave_dataset_address: str
    _dataset_rows: List[Dict[str, str]] = []
    _index: np.ndarray = None
    _model: SentenceTransformer = None

    def __init__(self, weave_dataset_address: str, model_name: str):
        super().__init__(
            weave_dataset_address=weave_dataset_address,
            model_name=model_name,
        )
        self._dataset_rows = [
            dict(row) for row in weave.ref(self.weave_dataset_address).get().rows
        ]
        self._model = SentenceTransformer(self.model_name)
        self._index = self._model.encode(
            sentences=[
                row["text"] + "\n\n# Image Descriptions\n" + row["image_descriptions"]
                for row in self._dataset_rows
            ],
            normalize_embeddings=True,
        )

    @weave.op()
    def search(self, query: str, top_k: int = 5):
        query_embeddings = self._model.encode([query], normalize_embeddings=True)
        scores = query_embeddings @ self._index.T
        sorted_indices = np.argsort(scores, axis=None)[::-1]
        top_k_indices = sorted_indices[:top_k].tolist()
        retrieved_pages = []
        for idx in top_k_indices:
            retrieved_pages.append(
                {
                    "text": self._dataset_rows[idx]["text"],
                    "image_descriptions": self._dataset_rows[idx]["image_descriptions"],
                    "source": self._dataset_rows[idx]["pdf_file"],
                    "dataset_idx": idx,
                }
            )
        return retrieved_pages

    @weave.op()
    def predict(self, query: str, top_k: int = 5):
        return self.search(
            "Generate a representation for this sentence that can be used to retrieve related articles:\n"
            + query,
            top_k,
        )
