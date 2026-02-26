from functools import partial
from os import path

from science_modeling_tools.common.inferencers.http_request_inferencer import HttpRequestInferencer
from rich_python_utils.io_utils.json_io import write_json
from rich_python_utils.datetime_utils.common import timestamp

debug_mode = True
logger = partial(write_json, file_path=path.join('.', '_logs', f'{timestamp()}.json'), append=True)

reasoner = HttpRequestInferencer(
    service_url='http://devvm20179.nha0.facebook.com:8087/generate',
    request_body_prompt_field_name='prompt_or_messages',
    model_id='claude-4-sonnet-genai',
    secret_key='mg-api-466441a8622f',
    max_retry=3,
    logger=logger,
    debug_mode=debug_mode
)

print(reasoner('Hello!'))