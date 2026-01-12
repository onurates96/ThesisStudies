def get_custom_prompt(user_story, diagram_type):
    # Temel yapı (Ortak alanlar)
    base_instruction = f"Generate a valid, syntactically correct MermaidJS {diagram_type}."
    
    # 1. SEQUENCE DIAGRAM İÇİN ÖZEL KURALLAR
    if diagram_type == "Sequence":
        rules = """
- Logic: Structure participants as Users -> Frontend -> Backend -> DB.
- Syntax: Use 'activate' and 'deactivate' for every request/response flow.
- Arrows: Use (->>) for requests and (-->>) for responses.
- Handling: Ensure all 'alt', 'else', and 'opt' blocks are covered.
- Start with: 'sequenceDiagram'
"""
    # 2. CLASS DIAGRAM İÇİN ÖZEL KURALLAR
    elif diagram_type == "Class":
        rules = """
- Logic: Focus on Entities, Attributes, and Relationships (1:1, 1:N).
- Syntax: Use '+' for public, '-' for private members.
- Relationships: Use '--' for association, '-->' for inheritance, and '*--' for composition.
- Start with: 'classDiagram'
"""
    # 3. STATE DIAGRAM İÇİN ÖZEL KURALLAR
    elif diagram_type == "State":
        rules = """
- Logic: Define states and events that trigger transitions.
- Syntax: Use '[*]' for initial and final states.
- Transitions: Use '-->' with descriptions for state changes.
- Start with: 'stateDiagram-v2'
"""
    # 4. COMPONENT/FLOW İÇİN ÖZEL KURALLAR
    else:
        rules = """
- Logic: Focus on system modules and directional flow.
- Syntax: Use 'graph TD' or 'graph LR'.
- Branding: Use meaningful labels inside nodes like [User Interface].
"""

    # Final Birleştirme
    full_prompt = f"""
{base_instruction}

Input Description: {user_story}

Specific Rules for this Diagram:
{rules}

Format: Respond ONLY with the MermaidJS code block. No conversation.
"""
    return full_prompt