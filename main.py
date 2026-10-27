"""Run with: uvicorn main:app --reload."""
# Uvicorn looks for the variable named app in this file.
# Routes are defined in api.py so this startup file stays short.
from src.ragaiml_mh2.api import app
