"""Colour constants for the few places that render custom HTML (source snippets, text tiles, answer boxes).

They mirror .streamlit/config.toml — Streamlit does not expose the active theme to Python, so when a colour changes
in the config it has to change here too.
"""

PRIMARY = "#1f4e8c"
TEXT = "#1c1c1c"
MUTED = "#5b6572"        # secondary text: labels, snippets, captions
BORDER = "#d9dde3"
SURFACE = "#ffffff"      # cards on the warm page background
SURFACE_ALT = "#eef2f7"  # matches secondaryBackgroundColor
