import base64
import io
import os
from glob import glob
from typing import Dict, List, Optional, Union

import weave
from pdf2image.pdf2image import convert_from_path
from PIL import Image
from PyPDF2 import PdfReader
from rich.progress import track

import wandb

from .llm_wrapper import MultiModalPredictor


class TextExtractionModel(weave.Model):
    documents_artifact_address: str
    predictor: MultiModalPredictor
    _artifact_dir: str

    def __init__(self, documents_artifact_address: str, predictor: MultiModalPredictor):
        super().__init__(
            documents_artifact_address=documents_artifact_address, predictor=predictor
        )
        api = wandb.Api()
        artifact = api.artifact(self.documents_artifact_address)
        self._artifact_dir = artifact.download()

    @weave.op()
    def extract_data_from_pdf_file(self, pdf_file: str, page_number: int) -> str:
        image = convert_from_path(
            pdf_file, first_page=page_number + 1, last_page=page_number + 1
        )[0]
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()
        img_base64 = base64.b64encode(img_bytes)
        img_base64 = img_base64.decode("utf-8")
        return f"data:image/png;base64,{img_base64}"

    @weave.op()
    def extract_from_pdf_file(self, pdf_file: str) -> List[Dict[str, str]]:
        extracted_document_pages = []
        with open(pdf_file, "rb") as file:
            reader = PdfReader(file)
            total_pages = len(reader.pages)
        pdf_file_base_name = pdf_file.split("/")[-1]
        for page_number in track(
            range(total_pages), description=f"Reading pages from {pdf_file_base_name}:"
        ):
            image = self.extract_data_from_pdf_file(pdf_file, page_number)
            page_text = self.predictor.predict(
                user_prompts=[
                    """Extract all the text present in the image in markdown format.
Make sure to format all mathematical notations and equations in latex format.""",
                    image,
                ]
            )
            image_descriptions = self.predictor.predict(
                system_prompt="""You are an expert physicist tasked with describing all the
images, diagrams, and figures present in a screenshot of a page from a physics textbook.

Here are a couple of rules you need to follow:

1. You are supposed to describe the images in a way that a visually impaired student can also
    understand.
2. While describing the images, diagrams, and figures, make sure to include all the important
    details and information present in them.
3. You will be provided with the text present in the image in markdown format (with formulae
    and mathematical notation represented in latex) to be used as context. You should refer to
    this text context to try to understand the images, diagrams, and figures in a more holistic
    and detailed manner.
4. You should pay special attention to the figure description of the respective image, diagram,
    or figure if it is present in the screenshot or the context.
""",
                user_prompts=[image],
            )
            extracted_document_pages.append(
                {
                    "text": page_text,
                    "image_descriptions": image_descriptions,
                    "pdf_file": pdf_file,
                    "page_number": page_number,
                }
            )
        return extracted_document_pages

    @weave.op()
    def predict(self, weave_dataset_name: Optional[str] = None) -> List[Dict[str, str]]:
        pdf_files = glob(os.path.join(self._artifact_dir, "keph10*.pdf")) + glob(
            os.path.join(self._artifact_dir, "keph20*.pdf")
        )
        all_extracted_document_pages = []
        for pdf_file in pdf_files:
            extracted_document_pages = self.extract_from_pdf_file(pdf_file)
            all_extracted_document_pages += extracted_document_pages
        if weave_dataset_name:
            weave.publish(
                weave.Dataset(
                    name=weave_dataset_name, rows=all_extracted_document_pages
                )
            )
        return all_extracted_document_pages


class PDFImageLoader(weave.Model):
    documents_artifact_address: str
    _artifact_dir: str

    def __init__(self, documents_artifact_address: str):
        super().__init__(documents_artifact_address=documents_artifact_address)
        api = wandb.Api()
        artifact = api.artifact(self.documents_artifact_address)
        self._artifact_dir = artifact.download()

    @weave.op()
    def extract_data_from_pdf_file(
        self, pdf_file: str, page_number: int
    ) -> Image.Image:
        image = convert_from_path(
            pdf_file, first_page=page_number + 1, last_page=page_number + 1
        )[0]
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()
        return Image.open(io.BytesIO(img_bytes))

    @weave.op()
    def extract_from_pdf_file(
        self, pdf_file: str
    ) -> List[Dict[str, Union[Image.Image, str, int]]]:
        extracted_document_pages = []
        with open(pdf_file, "rb") as file:
            reader = PdfReader(file)
            total_pages = len(reader.pages)
        for idx, page_number in track(
            enumerate(range(total_pages)), description="Reading pages:"
        ):
            image = self.extract_data_from_pdf_file(pdf_file, page_number)
            extracted_document_pages.append(
                {
                    "image": image,
                    "pdf_file": pdf_file,
                    "page_no": idx,
                }
            )
        return extracted_document_pages

    @weave.op()
    def predict(self, weave_dataset_name: Optional[str] = None) -> List[Dict[str, str]]:
        pdf_files = glob(os.path.join(self._artifact_dir, "keph10*.pdf")) + glob(
            os.path.join(self._artifact_dir, "keph20*.pdf")
        )
        all_extracted_document_pages = []
        for pdf_file in pdf_files:
            extracted_document_pages = self.extract_from_pdf_file(pdf_file)
            all_extracted_document_pages += extracted_document_pages
        if weave_dataset_name:
            weave.publish(
                weave.Dataset(
                    name=weave_dataset_name, rows=all_extracted_document_pages
                )
            )
        return all_extracted_document_pages
