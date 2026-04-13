def build_ai_next_turn_prompt_ch_style_simplified(
        turns: List[Dict[str, str]],  # New parameter
        character_profile: str = "",
        icebreaker: str = "",
        user_profile: str = ""
):
    # Build system prompt from persona info
    ch_system_prompt = """Conversation Guidance:
    Your name is {name}
    {description}
    Here are some facts about you: {instructions}
    """.format(
        name=name, description=description, instructions=instructions
    )

    # Add examples if provided
    examples_prompt = ""
    if len(dialog_examples) > 0:
        examples_prompt += "See the below dialogue examples of how to respond:\n"
        for example_dialog in dialog_examples:
            examples_prompt += "<user>:" + example_dialog["prompt"] + "\n"
            examples_prompt += "<model>:" + example_dialog["response"] + "\n"
        examples_prompt += "\n"
        ch_system_prompt += "\n" + examples_prompt

    # Initialize AI history with system prompt + historical conversation turns
    ai_history = [
        {"role": "user", "content": ch_system_prompt},
        {"role": "assistant", "content": ""},
    ]

    # Add icebreaker if provided
    if icebreaker:
        ai_history.append({"role": "user", "content": icebreaker})

    # Add historical conversation turns to AI history
    for turn in turns:
        # Map roles if needed (assistant -> assistant, user -> user, etc.)
        role = turn["role"]
        if role in ["user", "assistant", "system"]:
            ai_history.append({
                "role": role,
                "content": turn["content"]
            })

    return ai_history