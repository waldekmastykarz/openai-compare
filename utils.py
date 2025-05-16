from pathlib import Path
import prompty

def load_prompty(prompty_file_path: str) -> prompty.Prompty:
    p = Path(prompty_file_path).resolve().absolute()
    matter = prompty.load_prompty(p)
    attributes = matter["attributes"]
    content = matter["body"]
    attributes = prompty.Prompty.normalize(attributes, p.parent)
    prompt = prompty._load_raw_prompty(attributes, content, p, {})
    return prompt