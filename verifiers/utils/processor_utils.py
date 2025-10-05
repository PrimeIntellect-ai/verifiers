from verifiers.utils.image_utils import _base64_to_pil
from typing import Union, List, Dict, Any, TYPE_CHECKING
from inspect import signature

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase, ProcessorMixin

def supports_images(obj): # work as a replacement for  if isinstance(processing_class, ProcessorMixin), because we already type check with lazy importfrom inspect import signature
    if callable(obj):
        try:
            sig = signature(obj)
            return "images" in sig.parameters
        except TypeError:
            return False
    return False

def encode_chat_with_processor(
    conversation: List[Dict],
    processing_class: Union["PreTrainedTokenizerBase", "ProcessorMixin"],
    add_generation_prompt: bool = False,
    add_special_tokens: bool = False,
) -> tuple[list[int], Any, Any]:
    """
    Apply chat template and return token IDs, handling both tokenizer and processor.
    Supports base64-encoded images in the conversation.
    """

    if supports_images(processing_class): 
        prompt_text = processing_class.apply_chat_template(
            conversation=conversation,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
        images = []
        for msg in conversation:
            for c in msg.get("content", []):
                if c.get("type") == "image_url":
                    pil_img = _base64_to_pil(c["image_url"]["url"])
                    images.append(pil_img)

        inputs = processing_class(
            text=[prompt_text],
            images=images if images else None,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        return inputs["input_ids"][0].tolist(), inputs["image_grid_thw"][0].tolist(), inputs["pixel_values"].tolist()

    else:
        prompt_ids : List[int] = processing_class.apply_chat_template(
            conversation=conversation,
            add_generation_prompt=add_generation_prompt,
        )
        return prompt_ids,None,None
    
def encode_text_with_processor(
    text: str,
    processing_class: Union["PreTrainedTokenizerBase", "ProcessorMixin"],
) -> tuple[list[int], Any, Any]:
    """
    Encode plain text and return token IDs, handling both tokenizer and processor.
    """
    if supports_images(processing_class): 
        inputs = processing_class(
            text=[text],
            images=None,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"][0].tolist()
        image_grid = inputs.get("image_grid_thw", [None])[0].tolist()
        pixel_values = inputs.get("pixel_values", [None]).tolist()
        return input_ids, image_grid, pixel_values
    else:
        prompt_ids: list[int] = processing_class.encode(
            text
        )
        return prompt_ids, None, None