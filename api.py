"""
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import os
import cv2
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io
import uvicorn

from dolphin import DOLPHIN
from utils.utils import crop_margin, save_outputs, prepare_image, parse_layout_string, process_coordinates
from utils.markdown_utils import MarkdownConverter

app = FastAPI(
    title="DOLPHIN Document Recognition API",
    description="API for document and element recognition using DOLPHIN model",
    version="1.0.0",
)

# Initialize model
MODEL_PATH = os.getenv("MODEL_PATH", "./hf_model")
model = DOLPHIN(MODEL_PATH)

class RecognitionResult(BaseModel):
    label: str
    text: str
    bbox: Optional[List[float]] = None
    reading_order: Optional[int] = None


class PageRecognitionResponse(BaseModel):
    results: List[RecognitionResult]
    md_content: Optional[str] = None


class ElementRecognitionResponse(BaseModel):
    result: str
    recognition_result: List[RecognitionResult]
    md_content: Optional[str] = None


def process_elements(layout_results, padded_image, dims, model, max_batch_size=None):
    """Parse all document elements with parallel decoding"""
    layout_results = parse_layout_string(layout_results)

    # Store text and table elements separately
    text_elements = []  # Text elements
    table_elements = []  # Table elements
    figure_results = []  # Image elements (no processing needed)
    previous_box = None
    reading_order = 0

    # Collect elements to process and group by type
    for bbox, label in layout_results:
        try:
            # Adjust coordinates
            x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, previous_box = process_coordinates(
                bbox, padded_image, dims, previous_box
            )

            # Crop and parse element
            cropped = padded_image[y1:y2, x1:x2]
            if cropped.size > 0:
                if label == "fig":
                    # For figure regions, add empty text result immediately
                    figure_results.append(
                        {
                            "label": label,
                            "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                            "text": "",
                            "reading_order": reading_order,
                        }
                    )
                else:
                    # Prepare element for parsing
                    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    element_info = {
                        "crop": pil_crop,
                        "label": label,
                        "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                        "reading_order": reading_order,
                    }

                    # Group by type
                    if label == "tab":
                        table_elements.append(element_info)
                    else:  # Text elements
                        text_elements.append(element_info)

            reading_order += 1

        except Exception as e:
            print(f"Error processing bbox with label {label}: {str(e)}")
            continue

    # Initialize results list
    recognition_results = figure_results.copy()

    # Process text elements (in batches)
    if text_elements:
        text_results = process_element_batch(text_elements, model, "Read text in the image.", max_batch_size)
        recognition_results.extend(text_results)

    # Process table elements (in batches)
    if table_elements:
        table_results = process_element_batch(table_elements, model, "Parse the table in the image.", max_batch_size)
        recognition_results.extend(table_results)

    # Sort elements by reading order
    recognition_results.sort(key=lambda x: x.get("reading_order", 0))

    return recognition_results


def process_element_batch(elements, model, prompt, max_batch_size=None):
    """Process elements of the same type in batches"""
    results = []

    # Determine batch size
    batch_size = len(elements)
    if max_batch_size is not None and max_batch_size > 0:
        batch_size = min(batch_size, max_batch_size)

    # Process in batches
    for i in range(0, len(elements), batch_size):
        batch_elements = elements[i : i + batch_size]
        crops_list = [elem["crop"] for elem in batch_elements]

        # Use the same prompt for all elements in the batch
        prompts_list = [prompt] * len(crops_list)

        # Batch inference
        batch_results = model.chat_page(prompts_list, crops_list)

        # Add results
        for j, result in enumerate(batch_results):
            elem = batch_elements[j]
            results.append(
                {
                    "label": elem["label"],
                    "bbox": elem["bbox"],
                    "text": result.strip(),
                    "reading_order": elem["reading_order"],
                }
            )

    return results


@app.post("/recognize/page", response_model=PageRecognitionResponse)
async def recognize_page(file: UploadFile = File(...), max_batch_size: Optional[int] = 16):
    """
    Process a document page image and return recognition results

    Args:
        file: The image file to process
        save_dir: Optional directory to save results
        max_batch_size: Maximum batch size for processing elements

    Returns:
        PageRecognitionResponse containing recognition results and optional save path
    """
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Stage 1: Page-level layout and reading order parsing
        layout_output = model.chat_page("Parse the reading order of this document.", image)

        # Stage 2: Element-level content parsing
        padded_image, dims = prepare_image(image)
        recognition_results = process_elements(layout_output, padded_image, dims, model, max_batch_size)

        # Save outputs if save_dir is provided

        markdown_converter = MarkdownConverter()
        markdown_content = markdown_converter.convert(recognition_results)
        return PageRecognitionResponse(results=recognition_results, md_content=markdown_content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recognize/element", response_model=ElementRecognitionResponse)
async def recognize_element(file: UploadFile = File(...), element_type: str = "text"):
    """
    Process a single element image (text, table, formula) and return recognition results

    Args:
        file: The image file to process
        element_type: Type of element to process (text, table, formula)
        save_dir: Optional directory to save results

    Returns:
        ElementRecognitionResponse containing recognition result
    """
    try:
        # Validate element type
        if element_type not in ["text", "table", "formula"]:
            raise HTTPException(status_code=400, detail="element_type must be one of: text, table, formula")

        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = crop_margin(image)

        # Select appropriate prompt based on element type
        if element_type == "table":
            prompt = "Parse the table in the image."
            label = "tab"
        elif element_type == "formula":
            prompt = "Read text in the image."
            label = "formula"
        else:  # Default to text
            prompt = "Read text in the image."
            label = "text"

        # Process the element
        result = model.chat_element(prompt, image)

        # Create recognition result
        recognition_result = [
            {
                "label": label,
                "text": result.strip(),
            }
        ]

        # Save results if save_dir is provided

        # Save outputs if save_dir is provided
        markdown_converter = MarkdownConverter()
        markdown_content = markdown_converter.convert(recognition_result)


        return ElementRecognitionResponse(
            result=result, recognition_result=recognition_result, md_content=markdown_content
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
