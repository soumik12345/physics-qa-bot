import os
from glob import glob
from typing import Optional

import markdownify
import pdfplumber
from rich.progress import track

import wandb


class PDFConverter:
    def __init__(self, data_artifact_address: str):
        if wandb.run is None:
            api = wandb.Api()
            artifact = api.artifact(data_artifact_address)
            self.pdf_file = glob(os.path.join(artifact.download(), "*.pdf"))[0]
        else:
            artifact = wandb.use_artifact(data_artifact_address, type="dataset")
            self.pdf_file = glob(os.path.join(artifact.download(), "*.pdf"))[0]

    def convert_to_markdown(self, markdown_dir: str, max_pages: Optional[int] = None):
        image_dir = os.path.join(markdown_dir, "images")
        os.makedirs(os.path.join(markdown_dir, "images"), exist_ok=True)
        with pdfplumber.open(self.pdf_file) as pdf:
            all_pages = (
                pdf.pages
                if max_pages is None or len(max_pages) > len(pdf.pages)
                else pdf.pages[:max_pages]
            )
            for page_num, page in track(
                enumerate(all_pages), description="Converting PDF pages to Markdown"
            ):
                markdown_content = {}
                text = page.extract_text()
                markdown_content = {
                    "text": (
                        markdownify.markdownify(text, heading_style="ATX")
                        if text
                        else ""
                    ),
                    "images": [],
                }
                images = page.images
                image_counter = 1
                for image in images:
                    image_bbox = image["bbox"]
                    image_object = page.to_image()
                    cropped_image = image_object.crop(image_bbox)
                    image_filename = os.path.join(
                        image_dir, f"image_{image_counter}_page_{page_num}.png"
                    )
                    cropped_image.save(image_filename)
                    markdown_content["images"].append(image_filename)
                    image_counter += 1
                with open(os.path.join(markdown_dir, f"page_{page_num}.md"), "w") as f:
                    f.write(markdown_content["text"])
