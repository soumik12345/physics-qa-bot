import io
import base64

import weave
from openai import OpenAI
from pdf2image.pdf2image import convert_from_path


class TextExtractionModel(weave.Model):
    model_name: str
    _llm_client: OpenAI = None

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self._llm_client = OpenAI()

    @weave.op()
    def extract_data_from_pdf_file(self, pdf_file: str, page_number: int):
        images = []
        images_from_page = convert_from_path(
            pdf_file, first_page=page_number + 1, last_page=page_number + 1
        )
        for image in images_from_page:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_bytes = img_byte_arr.getvalue()
            img_base64 = base64.b64encode(img_bytes)
            img_base64 = img_base64.decode('utf-8')
            images.append(f"data:image/jpeg;base64,{img_base64}",)
        return images

    @weave.op()
    def predict(self, pdf_file: str):
        responses = []
        images = self.extract_data_from_pdf_file(pdf_file, 1)
        for image in images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
Extract all the text present in the image in markdown format.
Make sure to format all mathematical notations and equations in latex format.""",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image, "detail": "high"},
                        },
                    ],
                }
            ]
            responses.append(
                self._llm_client.chat.completions.create(model=self.model_name, messages=messages)
                .choices[0]
                .message
            )
        return responses
