"""
capture_screenshots.py
Captures screenshots of the running Streamlit app across all three tabs.
Saves to plots/screenshots/
"""

import os, time
from playwright.sync_api import sync_playwright

os.makedirs("plots/screenshots", exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 900})

    # ── 1. Home / Prediction tab (before predict) ─────────────────────────────
    page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
    time.sleep(4)  # let Streamlit fully render
    page.screenshot(path="plots/screenshots/01_prediction_tab_default.png", full_page=False)
    print("Saved: 01_prediction_tab_default.png")

    # ── 2. Trigger a prediction ────────────────────────────────────────────────
    # Click the Predict Price button (look for a button with that text)
    try:
        page.get_by_role("button", name="Predict Price").click()
        time.sleep(4)
        page.screenshot(path="plots/screenshots/02_prediction_result.png", full_page=False)
        print("Saved: 02_prediction_result.png")
    except Exception as e:
        print(f"Could not click Predict Price: {e}")
        page.screenshot(path="plots/screenshots/02_prediction_result.png", full_page=False)

    # ── 3. Model Metrics tab ───────────────────────────────────────────────────
    try:
        page.get_by_role("tab", name="Model Metrics").click()
        time.sleep(3)
        page.screenshot(path="plots/screenshots/03_model_metrics_tab.png", full_page=False)
        print("Saved: 03_model_metrics_tab.png")
    except Exception as e:
        print(f"Could not click Model Metrics tab: {e}")

    # ── 4. Data Explorer tab ───────────────────────────────────────────────────
    try:
        page.get_by_role("tab", name="Data Explorer").click()
        time.sleep(3)
        page.screenshot(path="plots/screenshots/04_data_explorer_tab.png", full_page=False)
        print("Saved: 04_data_explorer_tab.png")
    except Exception as e:
        print(f"Could not click Data Explorer tab: {e}")

    # ── 5. FastAPI Swagger UI ─────────────────────────────────────────────────
    page.goto("http://localhost:8000/docs", wait_until="networkidle", timeout=20000)
    time.sleep(2)
    page.screenshot(path="plots/screenshots/05_fastapi_swagger.png", full_page=False)
    print("Saved: 05_fastapi_swagger.png")

    browser.close()

print("\nAll screenshots saved to plots/screenshots/")
