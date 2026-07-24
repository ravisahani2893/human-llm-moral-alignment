import sys
from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:8000/"

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 1200})

    page.goto(BASE_URL)
    page.wait_for_selector("#batch-model-select input[type=checkbox]")
    page.uncheck('#batch-model-select input[data-role="all"]')
    page.check('#batch-model-select input[value="gemini"]')
    page.fill("#sample-size-input", "3")
    page.click("#run-batch-btn")
    page.wait_for_selector("#batch-status:has-text('Completed.')", timeout=120_000)
    page.wait_for_selector("#batch-ccc-wrap .ccc-block", timeout=15_000)

    page.locator("#batch-ccc-wrap").screenshot(path="/tmp/ccc_table_screenshot.png")
    print("saved screenshot")

    # Also dump the raw HTML of the table to check for any structural issue
    print(page.inner_html("#batch-ccc-wrap"))

    browser.close()
