# Day 01 — Mini Project
## AI Engineer CLI Profile

---

## Project Overview

**Project:** AI Engineer CLI Profile  
**Purpose:** A command-line tool that demonstrates your AI engineering knowledge  
**Difficulty:** Beginner  
**Time:** 2 hours  
**Code:** `07_code/03_ai_engineer_profile.py`

This is not a toy project. The structure and patterns used here — dataclasses, type hints, enum-based dispatch, clean CLI design — are the same patterns you will use in production AI systems.

---

## What You Build

A CLI application that:

1. Displays a complete AI engineering concept map
2. Explains the modern AI tech stack
3. Shows the 30-day learning roadmap
4. Provides side-by-side comparisons (RAG vs Fine-tuning, etc.)
5. Gives interview cheatsheets
6. Allows deep-dive into individual concepts

---

## Why This Project Matters

**Portfolio value:** Every AI Engineer interview will ask "what do you know about AI?" This project forces you to organize your knowledge in a structured, explainable format.

**Code quality:** The code demonstrates:
- Clean Python with type hints
- Dataclass-based data modeling
- Separation of data from presentation logic
- Proper CLI design

**Mental model:** Building this project forces you to understand every concept you define in it. If you can't write a clean one-liner definition, you don't understand it yet.

---

## Step-by-Step Instructions

### Step 1: Set Up Environment (15 minutes)

```bash
# Create day01 working directory
mkdir -p day01/07_code
cd day01/07_code

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install python-dotenv rich

# Verify
python --version  # Should be 3.11+
```

### Step 2: Run the Existing Code (10 minutes)

```bash
python 03_ai_engineer_profile.py
```

You should see the interactive menu. Explore every option.

### Step 3: Understand the Code Structure (20 minutes)

Read `03_ai_engineer_profile.py` carefully:

```python
# Data layer (what we store)
@dataclass
class AIConcept:
    name: str
    category: str
    one_liner: str
    ...

# Data (the actual knowledge)
AI_CONCEPTS: list[AIConcept] = [...]

# Display layer (how we show it)
def show_concept_map() -> None: ...
def show_tech_stack() -> None: ...

# Menu (user interaction)
def interactive_menu() -> None: ...
```

This separation of concerns — data, display, interaction — is fundamental to maintainable code.

### Step 4: Extend the Project (45 minutes)

Add these features to `03_ai_engineer_profile.py`:

**Extension 1: Add 5 more concepts** (10 minutes)
Add these to `AI_CONCEPTS`:
- Context Window
- Chunking
- Reranking
- Prompt Injection
- Guardrails

Each must have a proper definition, details, and use case.

**Extension 2: Add a quiz mode** (20 minutes)

```python
def quiz_mode(concepts: list[AIConcept], num_questions: int = 5) -> None:
    """
    Quiz the user on AI concepts.
    - Show the definition, ask for the name
    - OR show the name, ask for the definition
    - Track score
    """
    import random
    
    questions = random.sample(concepts, min(num_questions, len(concepts)))
    score = 0
    
    for i, concept in enumerate(questions, 1):
        print(f"\nQuestion {i}/{len(questions)}:")
        print(f"Category: {concept.category}")
        print(f"Definition: {concept.one_liner}")
        
        user_answer = input("What concept is this? ").strip().lower()
        correct = concept.name.lower()
        
        if user_answer == correct or user_answer in correct:
            print("✓ Correct!")
            score += 1
        else:
            print(f"✗ The answer was: {concept.name}")
    
    percentage = (score / len(questions)) * 100
    print(f"\nFinal Score: {score}/{len(questions)} ({percentage:.0f}%)")
    
    if percentage >= 90:
        print("Excellent! You know your AI concepts.")
    elif percentage >= 70:
        print("Good. Review the concepts you missed.")
    else:
        print("Keep studying. Read 02_concepts.md again.")
```

**Extension 3: Add a search function** (15 minutes)

```python
def search_concepts(query: str, concepts: list[AIConcept]) -> list[AIConcept]:
    """
    Search concepts by name, category, or definition.
    Returns all matching concepts.
    """
    query = query.lower()
    results = []
    
    for concept in concepts:
        searchable = (
            concept.name.lower()
            + concept.category.lower()
            + concept.one_liner.lower()
            + concept.details.lower()
        )
        if query in searchable:
            results.append(concept)
    
    return results
```

### Step 5: Add to Menu (10 minutes)

Update `interactive_menu()` to include:
- Option for quiz mode
- Option for concept search

### Step 6: Test Everything (10 minutes)

Run through every menu option. Verify:
- All options work without errors
- Quiz mode asks 5 questions and scores correctly
- Search returns relevant concepts
- Error handling works (press Ctrl+C, enter invalid option)

---

## Expected Output

```
╔══════════════════════════════════════════════════════════╗
║        AI ENGINEER KNOWLEDGE PROFILE — DAY 01            ║
║        30-Day AI Engineer Program                        ║
╚══════════════════════════════════════════════════════════╝

──────────────────────────────────────────────────
  MENU:
    1. Concept Map
    2. Tech Stack
    3. Learning Path
    4. Compare: RAG vs Fine-Tuning
    5. Compare: LLM vs Agent
    6. Compare: SQL vs Vector DB
    7. Interview Cheatsheet
    8. Concept Detail (LLM)
    9. Concept Detail (RAG)
    Q. Quiz Mode
    S. Search Concepts
    0. Exit

  Enter choice: 1
```

---

## Rubric

| Criterion | Points |
|-----------|--------|
| Code runs without errors | 20 |
| All original menu options work | 20 |
| 5 new concepts added (correct, detailed) | 20 |
| Quiz mode implemented and works | 20 |
| Search function implemented | 10 |
| Type hints throughout | 5 |
| No hardcoded strings (data in data structures) | 5 |
| **Total** | **100** |

---

## What Good Looks Like

**Excellent (90+):**
- All features work
- New concepts have thorough explanations
- Quiz mode tracks and shows score with feedback
- Search is case-insensitive and searches all fields
- Code is clean, typed, and follows the existing patterns

**Good (75-89):**
- All original features work
- Some new concepts added
- Quiz mode works

**Needs Revision (<75):**
- Errors on running
- Missing features
- New concepts have incomplete definitions

---

## Reflection Questions

After completing the project:

1. Could you add a new concept to this system in 2 minutes? (You should be able to.)
2. Can you explain every concept you added without looking at your notes?
3. Did the quiz reveal any concepts you thought you understood but didn't?

If you answered "no" to any of these, revisit the corresponding sections in `02_concepts.md`.

---

## Connection to Future Days

This project introduced:
- `@dataclass` — You'll use this heavily from Day 3+
- Type hints — Required from Day 2+
- CLI design — Pattern you'll extend in later projects
- Separation of data/logic — Core to all production code

On Day 9, you'll build an ML Prediction Service. The pattern is the same:
- Data models (dataclasses)
- Business logic (functions)
- API layer (FastAPI instead of CLI)

You've already built the foundation.
