import json
from pathlib import Path
from typing import Optional


class LanguageManager:
    def __init__(self, default_language: str = "en"):
        self.current_language = default_language
        self.translations = {}
        self.load_language(default_language)

    def load_language(self, language_code: str) -> bool:
        """
        Load translations from a JSON file for the given language code.
        Returns True if successful, False otherwise.
        """
        try:
            file_path = Path(__file__).parent.parent / f"locales/{language_code}.json"
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    self.translations = json.load(f)
                    self.current_language = language_code
                    return True
            else:
                # If language file doesn't exist, use empty translations (fallback to English)
                self.translations = {}
                return False
        except Exception:
            self.translations = {}
            return False

    def _(self, key: str, default: Optional[str] = None) -> str:
        """
        Get translation for a key. Returns the translation if found,
        otherwise returns the key itself or the default value if provided.
        """
        return self.translations.get(key, default if default is not None else key)

