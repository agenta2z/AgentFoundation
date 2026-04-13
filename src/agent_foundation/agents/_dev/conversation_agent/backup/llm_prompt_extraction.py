import json
from typing import List


def extract_historical_conversation_turns(df) -> List[List[Dict[str, str]]]:
    """Extract conversation histories with roles and content from llm_prompt chat_history."""
    all_conversations = []

    for idx, row in df.iterrows():
        if 'conversation_turns' in row and pd.notna(row['conversation_turns']):
            try:
                # Parse the JSON array in conversation_turns
                conversation_turns = json.loads(row['conversation_turns'])

                # Each item in conversation_turns should have an llm_prompt field
                for turn_idx, turn in enumerate(conversation_turns):
                    if 'llm_prompt' in turn and turn['llm_prompt']:
                        try:
                            # Parse the JSON string in llm_prompt field
                            llm_prompt_data = json.loads(turn['llm_prompt'])

                            # Extract chat_history array
                            if 'chat_history' in llm_prompt_data:
                                chat_history = llm_prompt_data['chat_history']

                                # Extract messages with both role and content
                                conversation = []
                                for message in chat_history:
                                    if 'role' in message and 'content' in message:
                                        conversation.append({
                                            'role': message['role'],
                                            'content': message['content']
                                        })

                                if conversation:  # Only add non-empty conversations
                                    all_conversations.append(conversation)
                            else:
                                print(f"No chat_history found in row {idx}, turn {turn_idx}")

                        except (json.JSONDecodeError, TypeError) as e:
                            print(f"Error parsing llm_prompt JSON in row {idx}, turn {turn_idx}: {e}")
                            continue

            except (json.JSONDecodeError, TypeError) as e:
                print(f"Error parsing conversation_turns JSON in row {idx}: {e}")
                continue

    return all_conversations


# Extract all prompts (content only, no roles)
historical_conversation_turns = extract_historical_conversation_turns(df_sql_user_threads_read)
print(historical_conversation_turns[0][0])