from os import path
from typing import Iterator

from rich_python_utils.string_utils.formatting.template_manager import TemplateManager

root_path = path.dirname(__file__)
prompt_path = path.join(root_path, 'prompts')

prompt_template = TemplateManager(templates=prompt_path)
inference_input = """```character
Meet Puss in Boots, the favorite fearless hero and swashbuckling star.
```"""
prompts: Iterator = prompt_template(template_key='master', inference_input=inference_input)

for prompt in prompts:
    print(prompt)
    input('press any key to continue')