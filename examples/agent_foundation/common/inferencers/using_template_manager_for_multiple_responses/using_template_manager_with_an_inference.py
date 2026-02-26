import json
from functools import partial
from os import path
from typing import Iterator

from science_modeling_tools.common.inferencers.http_request_inferencer import (
    HttpRequestInferencer,
)
from rich_python_utils.io_utils.json_io import write_json
from rich_python_utils.string_utils.formatting.template_manager import (
    TemplateManager,
)
from rich_python_utils.datetime_utils.common import timestamp
from rich_python_utils.string_utils.common import cut

def extract_last_json_block(response: str):
    json_str = cut(
        response,
        cut_before_last="```json",
        cut_after_last="```",
        keep_cut_before=False,
        keep_cut_after=False
    )
    return json.loads(json_str)["refined_query_list"]


root_path = path.dirname(__file__)
prompt_path = path.join(root_path, "prompts")

prompt_template = TemplateManager(templates=prompt_path)
debug_mode = True
logger = partial(
    write_json, file_path=path.join(".", "_logs", f"{timestamp()}.json"), append=True
)

reasoner = HttpRequestInferencer(
    service_url="http://devvm20179.nha0.facebook.com:8087/generate",
    request_body_prompt_field_name="prompt_or_messages",
    model_id="claude-4-sonnet-genai",
    secret_key="mg-api-466441a8622f",
    max_retry=3,
    logger=logger,
    debug_mode=debug_mode,
    response_post_processor=extract_last_json_block
)

inference_input = """```character
Meet Puss in Boots, the favorite fearless hero and swashbuckling star.
```"""
prompts: Iterator = prompt_template(
    template_key="master", inference_input=inference_input
)
response = reasoner(prompts, max_tokens=20480, temperature=0.7, top_p=0.9)

print(response)
