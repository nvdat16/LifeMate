"""
Gatekeeper Agent: Generic, Schema-Driven Out-of-Scope Triage.
Zero hardcoded domain keywords: Builds app-domain inverted index dynamically from the Catalog.
"""

import re
import unicodedata
from typing import Any, Dict, Set
from ...domain.scenario import ScenarioType
from ...domain.entities import SampleContext

VIETNAMESE_STOPWORDS = {
    "thì", "thôi", "giờ", "mới", "cho", "các", "những", "trong", "của", "khi",
    "nếu", "lại", "nên", "được", "không", "và", "này", "đó", "tại", "ra", "vào",
    "như", "để", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười"
}

def extract_meaningful_tokens(text: str) -> set[str]:
    """Extracts lowercase NFC tokens longer than 2 characters excluding stopwords."""
    if not text:
        return set()
    s_norm = unicodedata.normalize("NFC", text.lower())
    words = re.findall(r"[a-zà-ỹ0-9_]+", s_norm)
    return {w for w in words if len(w) >= 3 and w not in VIETNAMESE_STOPWORDS}

class GatekeeperAgent:
    """
    Dynamically classifies user requests by comparing request tokens against
    the catalog app definitions and installed apps context.
    """

    def __init__(self, catalog: dict[str, Any]):
        self.catalog = catalog
        self._app_domain_index = self._build_dynamic_catalog_index()

    def _build_dynamic_catalog_index(self) -> dict[str, set[str]]:
        """
        Dynamically constructs token vocabulary for every app in the catalog.
        Excludes high-frequency words across apps to ensure high specificity.
        """
        raw_app_words: dict[str, set[str]] = {}
        df: dict[str, int] = {}

        for app in self.catalog.get("apps", []):
            app_id = app.get("id", "")
            text = f"{app_id} {app.get('title', '')} {app.get('description', '')}"
            for t in app.get("tools", []):
                text += f" {t.get('name', '')} {t.get('description', '')}"

            tokens = extract_meaningful_tokens(text)
            raw_app_words[app_id] = tokens
            for w in tokens:
                df[w] = df.get(w, 0) + 1

        # Keep words that appear in at most 4 apps (domain-specific terms)
        specific_index: dict[str, set[str]] = {}
        for app_id, tokens in raw_app_words.items():
            specific_index[app_id] = {w for w in tokens if df.get(w, 1) <= 4}

        return specific_index

    def classify_intent(self, transcript: str, context: SampleContext) -> ScenarioType:
        """
        Determines if the request is In-Scope or Out-of-Scope based on
        installed apps and catalog capabilities.
        """
        u_tokens = extract_meaningful_tokens(transcript)
        installed_app_ids = set(context.installed_apps.keys())

        # Find which catalog apps are referenced by user query tokens
        referenced_catalog_apps = set()
        for app_id, app_tokens in self._app_domain_index.items():
            if u_tokens.intersection(app_tokens):
                referenced_catalog_apps.add(app_id)

        # 1. If user clearly asks for apps in the catalog, but NONE of them are installed -> OUT_OF_SCOPE
        if referenced_catalog_apps and referenced_catalog_apps.isdisjoint(installed_app_ids):
            return ScenarioType.OUT_OF_SCOPE

        # 2. If user query has zero relevance to installed tools -> Out of Scope / General chatter
        installed_tokens = set()
        for app_id in installed_app_ids:
            installed_tokens.update(self._app_domain_index.get(app_id, set()))

        if not u_tokens.intersection(installed_tokens):
            return ScenarioType.OUT_OF_SCOPE

        return ScenarioType.EXECUTE
