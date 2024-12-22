import pymupdf4llm, cv2, fitz, os, io
from pathlib import Path
from typing import List
from llama_parse import LlamaParse
from img2table.ocr import EasyOCR
from img2table.document import PDF, Image

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

class DocParser:
    def __init__(self, parser_name):
        self.parser_name= parser_name
        self.assets_dir= "./parsed_assets/"
        self.parser_function_map= {
            "LlamaParse": self.with_LlamaParse,
            "pymupdf4llm": self.with_pymupdf4llm
        }
        self.parsing_function= self.parser_function_map[parser_name]

        # Instantiation of OCR
        self.ocr = EasyOCR(lang=["en"])
                # Ensure the save directory exists
        os.makedirs(self.assets_dir, exist_ok=True)

    def parse(self, file_path):
        text_docs= self.parsing_function(file_path)
        if self.parser_name=="LlamaParse":
            self.extract_images(file_path)
        
        self.extract_tables(file_path)
        return text_docs
    
    def with_LlamaParse(self, file_path):
        print("LLamaParse is being used ...")
        parser = LlamaParse(result_type="markdown", verbose=False)
        data= parser.load_data(file_path=file_path)
        text_docs= [x.text for x in data]
        return text_docs
    
    def with_pymupdf4llm(self, file_path):
        #No need for standalone image extraction step, already done here
        output = pymupdf4llm.to_markdown(
            file_path, 
            write_images=True,
            image_path=self.assets_dir,
            extract_words= True,
            show_progress= False)

        text_docs= [x["text"].replace("-----", "") 
                    for x in output]
        return text_docs

    def extract_tables(self, file_path):
        # Instantiation of document, either an image or a PDF
        if Path(file_path).suffix==".pdf":
            doc = PDF(file_path)
        else:
            doc = Image(file_path)
        # Table extraction
        extracted_tables = doc.extract_tables(ocr=self.ocr,
                                            implicit_rows=True,
                                            implicit_columns=True,
                                            borderless_tables=True)
        
        margin= 20
        save_dir= Path(self.assets_dir)
        file_stem= Path(file_path).stem

        for p, (image, tables) in enumerate(
            zip(doc._images, 
                extracted_tables.values())):
            for i, t in enumerate(tables):
                table_image= image[t.bbox.y1-margin:t.bbox.y2+margin,
                                t.bbox.x1-margin:t.bbox.x2+margin]
                cv2.imwrite(save_dir.joinpath(f"{file_stem}_{p}_table{i}.png"), table_image)


    def extract_images(self, filepath):
        """
        Extracts images from the provided files and saves them to the specified directory.

        Args:
            files_to_process (List[str]): List of file paths to extract images from.
            save_dir (str): Directory to save the extracted images.

        Returns:
            None
        """
        # for filepath in files_to_process:
            # Open the document using PyMuPDF
        doc = fitz.open(filepath)
        save_dir= Path(self.assets_dir)

        for p in range(len(doc)):
            page = doc[p]

            # Iterate through images on the page
            for i, img in enumerate(page.get_images(), start=1):
                xref = img[0]  # Image reference ID

                # Extract image bytes
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Create a PIL Image object from the bytes
                pil_image = Image.open(io.BytesIO(image_bytes))

                # Save the image with a structured name
                image_name = f"{save_dir.joinpath(Path(filepath).stem)}_{p}_image{i}.png"
                pil_image.save(image_name)