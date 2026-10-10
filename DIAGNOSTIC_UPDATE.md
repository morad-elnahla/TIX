v6.2 diagnostic update

Fix: currency parsing accepts grouping spaces (including U+202F).
Adds search_trace logs: candidate products, catalog status, filtered products, ordered IDs, result limit and products sent to the model.
No product retrieval behavior was changed: the suspected accessory crowding still requires live evidence.
No live TIX/Groq validation was performed here.
Run Streamlit normally, start a new chat, ask for a phone under EGP 5000, and capture the response and search_trace terminal lines.
Do not include API keys when sharing logs.
