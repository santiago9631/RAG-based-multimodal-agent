from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from typing import List
import glob, time, base64, logging, uuid6
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.image_dir= "./parsed_assets/"
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0
        )

    @staticmethod
    def retry_with_delay(func, *args, delay=2, retries=30, **kwargs):
        """
        Helper method to retry a function call with a delay.
        """
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(delay)
        raise RuntimeError("Exceeded maximum retries.")

    def encode_image(self, image_path):
        """Getting the base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def image_summarize(self, img_base64):
        """Make image summary"""
        prompt = """You are an assistant tasked with summarizing images for retrieval. \
                    These summaries will be embedded and used to retrieve the raw image. \
                    Give a concise summary of the image that is well optimized for retrieval."""
        # chat = ChatGoogleGenerativeAI(model="gpt-4-vision-preview", max_tokens=1024)

        msg = self.llm.invoke(
            [HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        },]
                )])
        return msg.content

    def get_image_summaries(self):
                            # image_paths: List[str])->List[str]:
        """
        Generates summaries for a list of images using a generative AI model.

        Args:
            image_paths (List[str]): A list of file paths to images.

        Returns:
            List[str]: A list of textual summaries for each image.
        """
        image_summaries = []
        # for i, img_path in enumerate(image_paths):
        for img_path in sorted(glob.glob(f"{self.image_dir}*.png")):
            base64_image = self.encode_image(img_path)

            # img_base64_list.append(base64_image)
            # Append the AI-generated summary to the list
            image_summaries.append(
                self.retry_with_delay(self.image_summarize, base64_image)
                )
        return image_summaries

    def get_image_documents(self)->List[Document]:
        """
        Extracts images from files and generates corresponding text nodes with metadata.

        Args:
            files_to_process (List[str]): A list of file paths to extract images from.

        Returns:
            List[TextNode]: A list of nodes containing image summaries and metadata.
        """
        image_documents = []
        # Generate summaries for the extracted images
        image_summaries = self.get_image_summaries()
        image_paths= sorted(glob.glob(f"{self.image_dir}*.png"))

        for summary, image_path in zip(image_summaries, image_paths):
            # Append the created node to the list
            image_documents.append(
                Document(
                page_content=summary,
                metadata={"source": Path(image_path).name},
                id= str(uuid6.uuid6()),
                )
            )

        return image_documents
