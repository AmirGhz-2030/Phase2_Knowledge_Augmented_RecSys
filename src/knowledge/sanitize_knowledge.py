"""
Post-Processing Sanitizer for LLM Reasoning Outputs.
Extracts the clean synthesized paragraph from CoT deliberation traces.
"""

import os
import re
import json

def extract_pure_synthesis(text: str) -> str:
    if not text:
        return ""
    
    # 0. Clean common LLM metadata and thought preambles
    text = re.sub(r"^Here'?s (?:a )?(?:thinking|thought) process[^\n:]*:\s*\*?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"^Thinking Process:\s*\*?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"^Thinking Process:.*?(?=\n\n|\Z)", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^\s*(\*+\s*)?Attempt\s*\d+\s*\(Mental Draft\):?", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"^[\*\-\•\s]+", "", text).strip()
    
    # 1. Look for explicit synthesis / polish markers in the CoT trace
    patterns = [
        r"(?:Initial thought synthesis|Drafting Check|Final Polish|Synthesize the Profile|Draft 1 \(Focus on premise/tone\)|Refine and Polish)[^\n:]*:\s*\*?\s*(.*?)(?=\s*\*\s*\d|\s*\*\s*2-3|\s*Ensure|\Z)",
        r"(?:Sentence 1[^\n:]*:\s*)(.*?)(?=\s*\*\s*\d|\s*\*\s*2-3|\Z)",
        r"(?:Drafting Sentences\s*:\s*)(.*?)(?=\s*\*\s*\d|\s*\*\s*2-3|\Z)"
    ]
    
    for pat in patterns:
        m = re.search(pat, text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            # Clean inner bullet points or markers
            candidate = re.sub(r"\*\s*\*[A-Za-z0-9\s/]+:\*\*", "", candidate)
            candidate = re.sub(r"\s+", " ", candidate).strip(" *-\n\"")
            if len(candidate) > 40 and not candidate.startswith("Analyze"):
                return candidate

    # 2. If no pattern matched, strip numbered steps and return the concluding sentences
    cleaned = re.sub(r"^\s*\d+\.\s+\*\*?[A-Za-z\s/,\(\)]+:\*\*?", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"The user wants a concise[^\n]*\.", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\*\s*\*[A-Za-z0-9\s/]+:\*\*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" *-\n\"")
    
    # Find complete sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned) if len(s.strip()) > 15 and not s.strip().startswith("Analyze")]
    if sentences:
        return " ".join(sentences[:3])
    
    return cleaned


def clean_existing_knowledge_files():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    knowledge_dir = os.path.join(base_dir, "data", "knowledge")
    
    # 1. Clean movies knowledge
    m_path = os.path.join(knowledge_dir, "gemma_item_knowledge.json")
    if os.path.exists(m_path):
        with open(m_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        
        cleaned_items = {}
        for k, v in items.items():
            pure_text = extract_pure_synthesis(v)
            if not pure_text or len(pure_text) < 20:
                pure_text = v
            cleaned_items[k] = pure_text
            
        with open(m_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_items, f, indent=2, ensure_ascii=False)
        print(f"Cleaned {len(cleaned_items)} movies in {m_path}")

    # 2. Clean users knowledge
    u_path = os.path.join(knowledge_dir, "gemma_user_reasoning.json")
    if os.path.exists(u_path):
        with open(u_path, "r", encoding="utf-8") as f:
            users = json.load(f)
            
        cleaned_users = {}
        for k, v in users.items():
            pure_text = extract_pure_synthesis(v)
            if not pure_text or len(pure_text) < 20:
                pure_text = v
            cleaned_users[k] = pure_text
            
        with open(u_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_users, f, indent=2, ensure_ascii=False)
        print(f"Cleaned {len(cleaned_users)} users in {u_path}")


if __name__ == "__main__":
    clean_existing_knowledge_files()
