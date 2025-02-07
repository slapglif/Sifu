from typing import List, Dict, Any

def get_clean_message_list(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Clean and format a list of messages for the model.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        List of cleaned message dictionaries
    """
    cleaned_messages = []
    for message in messages:
        if not isinstance(message, dict):
            continue
            
        role = message.get('role', '').strip().lower()
        content = message.get('content', '')
        
        if not role or not content:
            continue
            
        cleaned_messages.append({
            'role': role,
            'content': content
        })
        
    return cleaned_messages 