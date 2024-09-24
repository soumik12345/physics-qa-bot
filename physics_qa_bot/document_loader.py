import base64
import io
from typing import Optional

import weave
from pdf2image.pdf2image import convert_from_path
from PyPDF2 import PdfReader
from rich.progress import track

from .llm_wrapper import MultiModalPredictor


class TextExtractionModel(weave.Model):
    predictor: MultiModalPredictor

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
    def predict(self, pdf_file: str):
        extracted_document_pages = []
        with open(pdf_file, "rb") as file:
            reader = PdfReader(file)
            total_pages = len(reader.pages)
        for page_number in track(range(total_pages), description="Reading pages:"):
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
                }
            )
        return extracted_document_pages
