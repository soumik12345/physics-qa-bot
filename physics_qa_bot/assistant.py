from typing import List, Optional

import weave
from PIL import Image

from .llm_wrapper import MultiModalPredictor
from .utils import base64_encode_image


class PhysicsQAAssistant(weave.Model):
    multi_modal_query_predictor: MultiModalPredictor
    math_query_predictor: MultiModalPredictor
    text_query_predictor: MultiModalPredictor
    retriever: weave.Model
    weave_dataset_address: str

    def __init__(
        self,
        multi_modal_query_model: str,
        math_query_model: str,
        text_query_model: str,
        retriever: weave.Model,
        weave_dataset_address: str,
    ):
        super().__init__(
            multi_modal_query_predictor=MultiModalPredictor(
                model_name=multi_modal_query_model
            ),
            math_query_predictor=MultiModalPredictor(model_name=math_query_model),
            text_query_predictor=MultiModalPredictor(model_name=text_query_model),
            retriever=retriever,
            weave_dataset_address=weave_dataset_address,
        )

    @weave.op()
    def add_image_descriptions_to_query(
        self, query: str, images: Optional[List[Image.Image]] = None
    ) -> str:
        if images is not None:
            image_descriptions = self.multi_modal_query_predictor.predict(
                user_prompts=[
                    base64_encode_image(image_path=image) for image in images
                ],
                system_prompt="""You are an expert physicist tasked with describing all the
        images, diagrams, and figures present in a screenshot of a page from a physics textbook.

        Here are a couple of rules you need to follow:

        1. You are supposed to describe the images in a way that a visually impaired student can also
            understand.
        2. While describing the images, diagrams, and figures, make sure to include all the important
            details and information present in them.""",
            )
            query = f"""
        ---
        {query}
        ---

        You are to answer the question based on the provided description of images:

        ---
        {image_descriptions}.
        ---
        """
        return query

    @weave.op()
    def add_retrieved_context_to_query(self, query: str, top_k: int = 5):
        retrieved_pages = self.retriever.predict(query, top_k=top_k)
        context = []
        for page in retrieved_pages:
            refined_text = self.text_query_predictor.predict(
                system_prompt="""You are a helpful and experienced physics educator.
            Provided with the text from a physics textbook, you are to refine it by removing any questions or
            exercises that might be present in the text. Answer just with the refined text and nothing else.
            Make sure to not alter the text other than any questions or exercises.""",
                user_prompts=page["text"],
            )
            context.append(
                "---\n"
                + refined_text
                + "\n\n# Image Descriptions"
                + page["image_descriptions"]
                + "\n---"
            )
        context = "\n\n".join(context)
        return f"""
        # You are to answer the following question based on the provided context:
        {query}
        
        # Provided context:
        {context} 
        """

    @weave.op()
    def answer_query(
        self, query: str, language: str = "english", is_query_problem: bool = False
    ) -> str:
        nature_of_question = "problem" if is_query_problem else "question"
        system_prompt = f"""
        You are an expert physicist tasked with answering a physics {nature_of_question}.
        You are to think step-by-step regarding how the question can be answered provide
        a detailed and complete answer based on the provided context which consist of
        relevant source from a physics textbook.

        Here are some rules you need to follow:

        1. You are to respond in the following language: {language}
        2. You should try to justify your answer theoretically and mathematically whenever possible.
        3. The question may be accompanied by detailed descriptions of images, diagrams, and figures.
            You should pay close attention to the these descriptions of images, diagrams, and figures.
        4. You should pay close attention to the provided context and answering the question.
        5. If the context contains contains any questions, you should ignore them and focus on answering
            the main question by paying close attention to the rest of the context provided.
        6. If you are asked to define certain terminologies, you should provide a detailed definition
            supported by examples.
        """
        return (
            self.math_query_predictor.predict(
                user_prompts=[query],
                system_prompt=system_prompt,
            )
            if is_query_problem
            else self.text_query_predictor.predict(
                user_prompts=[query],
                system_prompt=system_prompt,
            )
        )

    @weave.op()
    def predict(
        self,
        query: str,
        images: Optional[List[Image.Image]] = None,
        language: str = "english",
        is_query_problem: bool = False,
        top_k: int = 5,
    ) -> str:
        query = self.add_image_descriptions_to_query(query, images)
        retrieval_augmented_query = self.add_retrieved_context_to_query(query, top_k)
        return self.answer_query(retrieval_augmented_query, language, is_query_problem)
