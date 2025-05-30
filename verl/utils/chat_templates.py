import hashlib
from pathlib import Path

chat_template_folder = Path(__file__).resolve().parent / "training_chat_templates"


def hash_chat_template(chat_template: str) -> str:
    sha256_hash = hashlib.sha256()
    sha256_hash.update(chat_template.encode("utf-8"))
    return sha256_hash.hexdigest()


# Run hash_chat_template(AutoTokenizer.from_pretrained("<model name>").chat_template) to get hash for new models' chat templates
chat_template_replacement = {
    # Qwen/Qwen3 chat template hash
    "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8": (chat_template_folder / "qwen3.jinja2").read_text(),
    # Qwen/QwQ-32B chat template hash
    "16488ad69d85833e5f66c50174f39834b688343337e9337f018deaa21b354117": (chat_template_folder / "qwq.jinja2").read_text(),
}
