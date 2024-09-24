from typing import Dict, List, Union, cast

import bm25s
import torch
import weave
from colpali_engine import ColPali, ColPaliProcessor
from PIL import Image


class BM25Retriever(weave.Model):
    weave_dataset_address: str
    _corpus: List[Dict[str, str]] = []
    _index: bm25s.BM25 = None

    def __init__(self, weave_dataset_address: str):
        super().__init__(weave_dataset_address=weave_dataset_address)
        self._index = bm25s.BM25()
        dataset_rows = weave.ref(weave_dataset_address).get().rows
        self._corpus = [dict(row) for row in dataset_rows]
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
                }
            )
        return output

    @weave.op()
    def predict(self, query: str, top_k: int = 5):
        return self.search(query, top_k)


class ColPaliRetriever(weave.Model):
    model_name: str
    processor_name: str
    device_map: str
    weave_dataset_address: str
    _corpus: List[Dict[str, Union[Image.Image, str]]] = []
    _index: torch.Tensor = None
    _model: ColPali = None
    _processor: ColPaliProcessor = None

    def __init__(
        self,
        weave_dataset_address: str,
        model_name: str,
        processor_name: str,
        device_map: str = "cuda",
    ):
        super().__init__(
            model_name=model_name,
            processor_name=processor_name,
            device_map=device_map,
            weave_dataset_address=weave_dataset_address,
        )
        self._model = cast(
            ColPali,
            ColPali.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
            ),
        )
        self._processor = cast(
            ColPaliProcessor, ColPaliProcessor.from_pretrained(self.processor_name)
        )
