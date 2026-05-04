"""Magic-string constants for the template subsystem.

Centralizes the strings that travel between YAML and Python: template
root spaces (directory namespaces under ``prompt_templates/``), variant
variable names, variant identifiers, template_key values, and field
names on ``TemplatedInferencerBase``.

Importing these instead of spelling the strings inline avoids typos and
lets renames be one-shot. Used by ``template_defaults.py`` (where the
named ``InferencerTemplateDefaults`` constants are constructed) and by
any orchestrator/test code that manipulates template fields by name.
"""

# === Template root spaces (directories under prompt_templates/) ===
SPACE_TASK_BREAKDOWN = "task_breakdown"
SPACE_PLAN = "plan"
SPACE_IMPLEMENTATION = "implementation"
SPACE_DEEP_RESEARCH = "deep_research"
SPACE_CONVERSATION = "conversation"

# === Template variant variable names (keys in template_variables dict) ===
VAR_TASK_PREAMBLE = "task_preamble"
VAR_TASK_INSTRUCTIONS = "task_instructions"
VAR_TASK_RESPONSE_FORMAT = "task_response_format"

# === Template variant value names (variant identifiers — _variables/<var>/<value>.jinja2) ===
VARIANT_AGGREGATION = "aggregation"
VARIANT_DEFAULT = "default"

# === Template key names (main template selector) ===
KEY_INITIAL = "initial"
KEY_REVIEW = "review"
KEY_FOLLOWUP = "followup"

# === TemplatedInferencerBase field names (dict keys when manipulating YAML nodes) ===
FIELD_TEMPLATE_ROOT_SPACE = "template_root_space"
FIELD_TEMPLATE_KEY = "template_key"
FIELD_TEMPLATE_VARIABLES = "template_variables"
FIELD_TEMPLATE_EXTRA_FEED = "template_extra_feed"
FIELD_TEMPLATE_VERSION = "template_version"
