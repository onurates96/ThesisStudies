def get_custom_prompt(user_story, diagram_type):
    base_instruction = f"Generate a valid, syntactically correct MermaidJS {diagram_type} diagram."

    output_contract = """
Output Contract (must follow exactly):
- Return ONLY ONE code block.
- The code block MUST start with ```mermaid and end with ```.
- The FIRST non-empty line inside the block MUST be the correct diagram header:
- Do NOT add any explanation text outside the code block.
- Do NOT return multiple code blocks.
""".strip()

    # Robust template registry (all examples are inside code)
    TEMPLATES = {
        "Sequence": {
            "rules": """
- Logic: Structure participants as User -> Frontend -> Backend -> DB.
- Syntax: Use activate/deactivate for every request/response flow.
- Arrows: Use (->>) for requests and (-->>) for responses.
- Handling: Ensure all alt/else/opt blocks are properly closed.
- Start with: sequenceDiagram

Examples: Example (format reference only):

sequenceDiagram
participant U as User
participant FE as Frontend
participant BE as Backend
participant DB as Database

U->>FE: Submit request
activate FE
FE->>BE: POST /api/action
activate BE
BE->>DB: Insert/Query
activate DB
DB-->>BE: OK
deactivate DB
BE-->>FE: 200 Success
deactivate BE
FE-->>U: Success message
deactivate FE

""".strip(),
        },
        "Class": {
            "rules": """
You are generating a Mermaid UML class diagram.

Hard constraints (must follow exactly):
1) Output ONLY a single Mermaid code block.
2) The FIRST non-empty line inside the code block MUST be: classDiagram
3) Class declarations MUST use a space: class ClassA { ... }
4) After declaring class Name, you must either open a {} block immediately or define members using the Name : member format — member lines can never start standalone with + or -.
5) Use ONLY these relationship connectors (single standard, do not use reversed forms):
   Inheritance: <|--
   Composition: *--
   Aggregation: o--
   Association: -->
   Solid link: --
   Dependency: ..>
   Realization: ..|>
   Dashed link: ..
6) Relationship label goes ONLY at the end after a colon:
   A --> B : label
   Do NOT put quoted words between connector and class name.
7) Cardinality/multiplicity:
   - Use ONLY on relationship lines.
   - MUST be quoted and placed next to class names:
     A "1" --> "0..*" B : label
   - Allowed multiplicities ONLY: "1", "0..1", "1..*", "*", "n", "0..n", "1..n"   

8) Forbidden patterns:
   - subclassdiagram (never output)
   - class X {"1"} (never put cardinality inside class definition blocks)
   - invalid tokens like "1..^"
   - quoted arrows like "-->" or "<--"
   - No comments (//, #, /* */) anywhere.
   - No bullet lists (-, *) anywhere.
   - Do NOT output JSON-like objects inside diagrams.
9) Member syntax:   
   - Attributes have NO parentheses. Methods HAVE ().
   - Visibility prefixes: + (public), - (private), # (protected), ~ (internal)
   - Optional return type comes AFTER ')', separated by a space: login() bool
   - Optional generic types: use ~Type~ (e.g., List~int~)

Examples:

Example 1 — Animal example:
---
title: Animal example
---
classDiagram    
    Animal <|-- Duck    
    Animal <|-- Fish
    Animal <|-- Zebra
    Animal : +int age
    Animal : +string gender
    Animal : +isMammal() bool
    Animal : +mate() void
    class Duck{
        +string beakColor
        +swim() void
        +quack() void
    }
    class Fish{
        -int sizeInFeet
        -canEat() bool
    }
    class Zebra{
        +bool is_wild
        +run() void
    }

Example 2 — Bank example (Attributes and Operations):
---
title: Bank example
---
classDiagram
    class BankAccount
    BankAccount : +string owner
    BankAccount : +decimal balance
    BankAccount : +deposit(amount:decimal) void
    BankAccount : +withdrawal(amount:decimal) void

Example 3 — Defining Relationships (single standard connectors):
classDiagram
classA <|-- classB : Inheritance
classC *-- classD : Composition
classE o-- classF : Aggregation
classG --> classH : Association
classI -- classJ : Link(Solid)
classK ..> classL : Dependency
classM ..|> classN : Realization
classO .. classP : Link(Dashed)

Example 4 — Cardinality / Multiplicity on relations:
classDiagram
    Customer "1" --> "*" Ticket
    Student "1" --> "1..*" Course
    Galaxy "1" --> "*" Star : contains
""".strip(),
        },
        "State": {
            "rules": """
- Logic: Define states and events that trigger transitions.
- Syntax: Use '[*]' for initial and final states.
- Transitions: Use '-->' with optional labels for state changes.
- Start with: stateDiagram-v2

Example": Example:

stateDiagram-v2
[*] --> Idle
Idle --> Processing: start
Processing --> Approved: approve
Processing --> Rejected: reject
Approved --> [*]: finish
Rejected --> [*]: finish

""".strip(),
        },
        "Action/Flow": {
            "rules": """
- Logic: Focus on system modules and directional flow.
- Syntax: Use 'flowchart TD' (preferred) or 'graph TD/LR'.
- Use meaningful labels inside nodes like [User Interface].
- Start with: flowchart TD

Example": Example:

flowchart TD
U[User] --> UI[User Interface]
UI --> API[Backend API]
API --> DB[(Database)]
API --> PAY[Payment Service]
PAY --> API
DB --> API
API --> UI
UI --> U
""".strip(),
        },
    }

    # Normalize diagram_type (in case you pass "Other" etc.)
    normalized = str(diagram_type).strip()
    if normalized not in TEMPLATES:
        # Map common fallbacks to Action/Flow
        normalized = "Class"

    tpl = TEMPLATES[normalized]
    rules = tpl.get("rules", "")

    prompt = f"""
    {base_instruction}

    {output_contract}

    Input Description:
    {user_story}

    Specific Rules for this Diagram:
    {rules}

    """.strip()

    return prompt

