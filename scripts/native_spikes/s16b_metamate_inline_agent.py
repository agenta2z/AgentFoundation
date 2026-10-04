"""S16b — Metamate 2.0 SDK: an inline agent config with
``system_prompt_mode="append"`` as the per-request carrier of session
instructions (the S16 ``internal_prompt`` path is treated by the model as
pasted, untrusted text).

Questions:
  1. Does the model follow instructions appended to its system prompt?
  2. Does it use host state (SOP catalog / active phase) given that way?
  3. Does a conversation resume by id (remember an earlier fact)?
  4. Does a changed append on a later request take effect?

    buck2 run @//mode/dbgo //_tony_dev/CoreProjects/AgentFoundation/scripts/native_spikes:s16b_metamate_inline_agent
"""

from __future__ import annotations

import asyncio

from agent_foundation.common.inferencers.agentic_inferencers.external.metamate.common import (
    DEFAULT_API_KEY,
    DEFAULT_SURFACE,
)
from msl.metamate.sdk.client import MetamateSDKClient
from msl.metamate.sdk.types import (
    MetamateAgentConfig,
    MetamateAgentProviderType,
    MetamateInlineAgentConfig,
    MetamateLLMVMConfig,
    MetamateOrchestration,
    MetamateOrchestrationType,
    MetamateSDKClientConfig,
    MetamateSDKInput,
    MetamateSessionConfig,
)

_RULES = """## How this session works
You are working inside the AgentFoundation host application, which runs
Standard Operating Procedures (SOPs). End EVERY reply with the exact token
{token}.

## Available SOPs
- Model Optimization (`model_optimization`)
- Code Optimization (`code_optimization`)

## Current state
Active SOP: Model Optimization, phase 0a "Setup workflow target path". Next
step: ask the user for the workflow target path."""


def _orchestration(prompt: str) -> MetamateOrchestration:
    return MetamateOrchestration(
        type=MetamateOrchestrationType.LLMVM,
        llmvm_config=MetamateLLMVMConfig(
            entry="Unified Auto",
            agent_config=MetamateAgentConfig(
                path_or_id="",
                provider_type=MetamateAgentProviderType.INLINE,
                inline_config=MetamateInlineAgentConfig(
                    name="agentfoundation-host",
                    description="Conversation host running AgentFoundation SOPs.",
                    metamate_api_key=DEFAULT_API_KEY,
                    prompt=prompt,
                    system_prompt_mode="append",
                ),
            ),
        ),
    )


async def _turn(text: str, prompt: str, conversation_id: str | None) -> tuple[str, str]:
    final, cid = "", conversation_id
    async for event in MetamateSDKClient.execute_and_stream(
        MetamateSDKInput(text=text),
        MetamateSDKClientConfig(api_keys=[DEFAULT_API_KEY], surface=DEFAULT_SURFACE),
        _orchestration(prompt),
        session_config=MetamateSessionConfig(conversation_id=conversation_id)
        if conversation_id
        else None,
    ):
        cid = event.conversation_id or cid
        if event.is_complete:
            final = event.text
    return final, cid or ""


async def amain() -> None:
    text, cid = await _turn(
        "Remember the codeword PELICAN-42. Which SOPs can you run, and which is active?",
        _RULES.format(token="[ZETA-END]"),
        None,
    )
    print("T1:", text[-500:])
    print("Q1 follows appended instructions:", text.rstrip().endswith("[ZETA-END]"))
    print(
        "Q2 uses host state:",
        "Model Optimization" in text and "Code Optimization" in text,
    )
    text, cid = await _turn(
        "What codeword did I give you? Reply with just the codeword.",
        _RULES.format(token="[OMEGA-END]"),
        cid,
    )
    print("T2:", text[-300:])
    print("Q3 resumed conversation:", "PELICAN-42" in text)
    print("Q4 changed append takes effect:", "[OMEGA-END]" in text)
    print("conversation:", cid)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
