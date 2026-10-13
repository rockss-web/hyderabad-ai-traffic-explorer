# Hyderabad AI Traffic Explorer

## Run on Windows
1. Extract this ZIP.
2. Open the extracted folder in File Explorer.
3. Click the address bar, type `cmd`, press Enter.
4. Run each command:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
5. Open the local URL printed in the terminal (usually http://localhost:8501).

## Important limitations
This is an educational prototype. It uses synthetic training data and illustrative road corridors/coordinates. The coloured lines are model-generated demo estimates, not current Google Maps traffic. The base map needs internet access. Do not use this for real-world navigation.

## Online deployment
Upload `app.py`, `requirements.txt`, and `README.md` to a GitHub repository, then deploy it at https://share.streamlit.io/ with `app.py` as the main file.
