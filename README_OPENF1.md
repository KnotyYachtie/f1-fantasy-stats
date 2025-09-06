# OpenF1 Integration (Patch)

This patch adds OpenF1 ingestion to the app.

## Files
- requirements.txt: adds httpx
- openf1_client.py: HTTP client
- data_source.py: converts OpenF1 data to the app's sessions/results schema
- app_patch_snippet.txt: code to paste into your app.py (imports + sidebar)

## Deploy
1) Add these files to your repo root.
2) Update your app.py using the snippet in app_patch_snippet.txt to add the sidebar radio and fetch button.
3) Commit -> Streamlit redeploys.
