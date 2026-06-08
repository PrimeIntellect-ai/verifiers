import base64
import textwrap
from os import PathLike
import os
import re
import inspect


def normalize_url_markdown(url: str) -> str:
    """Process URLs extracted from markdown, remove escape characters"""

    # Remove leading and trailing whitespace
    url = url.strip()

    # Remove escape backslashes before common markdown characters
    url = re.sub(r"\\([_()[\]*#!&?])", r"\1", url)

    return url


def text_dedent(multi_line_str: str) -> str:
    """
    abbreviation for removing superfluous start-of-line indenting from multi-line strings
    :param multi_line_str: a string value from a multi-line string expression
    :return: the multi-line string with any start-of-line whitespace that all lines have removed,
                plus any starting and ending newlines removed
    """
    return textwrap.dedent(multi_line_str).strip()


def strip_extension(filename):
    """
    Removes the file extension from a filename or file path.

    Args:
        filename (str): The file name or path.

    Returns:
        str: The file name or path without the extension.
    """
    return os.path.splitext(filename)[0]


def encode_image(image_path: str | PathLike) -> str:
    """
    credit to OpenAI docs
    :param image_path: path of image file to convert to base-64-encoded string
    :return: a base-64-encoded string version of the image file
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_image_buffer(buffer: bytes) -> str:
    """
    credit to OpenAI docs
    :param image_path: path of image file to convert to base-64-encoded string
    :return: a base-64-encoded string version of the image file
    """
    return base64.b64encode(buffer).decode("utf-8")


def _get_doc_from_frame(frame):
    co = frame.f_code
    name = co.co_name
    func = frame.f_globals.get(name)
    if (inspect.isfunction(func) or inspect.ismethod(func)) and func.__doc__:
        return inspect.getdoc(func)
    self_obj = frame.f_locals.get("self")
    if self_obj:
        cls = type(self_obj)
        meth = getattr(cls, name, None)
        if (inspect.isfunction(meth) or inspect.ismethod(meth)) and meth.__doc__:
            return inspect.getdoc(meth)
    consts = co.co_consts
    if consts and isinstance(consts[0], str):
        return consts[0]
    return None


def extract_doc_description(doc: str) -> str:
    """
    Given a full docstring, return only the desc part,
    i.e. all lines up until the first section header like
    'Parameters:', 'Returns:', etc.
    """
    if not doc:
        return ""
    lines = doc.splitlines()
    desc_lines = []
    section_rx = re.compile(r"^(?:Args?|Parameters?|Returns?|Yields?|Raises?):")
    for line in lines:
        if section_rx.match(line):
            break
        desc_lines.append(line)
    # strip leading/trailing blank lines, then re‑join
    return "\n".join(desc_lines).strip()


def extract_doc_description_from_frame(frame) -> str:
    """
    Given a frame object, return the desc part of the docstring
    of the function or method that the frame is in.
    """
    doc = _get_doc_from_frame(frame)
    return extract_doc_description(doc)
