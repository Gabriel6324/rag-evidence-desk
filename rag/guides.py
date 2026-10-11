"""Shared field guidance for the browser and desktop configuration window."""
import json
from .config import ROOT


def load_guides():
    return json.loads((ROOT / 'web/help-content.json').read_text(encoding='utf-8'))
