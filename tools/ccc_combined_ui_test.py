"""Sample test: 10 scenarios, 2 models, verify the Combined CCC row renders in the batch panel."""
import sys
from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:8000/"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))

        page.goto(BASE_URL)
        page.wait_for_selector("#batch-model-select input[type=checkbox]")
        page.uncheck('#batch-model-select input[data-role="all"]')
        page.check('#batch-model-select input[value="gemini"]')
        page.check('#batch-model-select input[value="lama"]')
        page.fill("#sample-size-input", "10")
        page.click("#run-batch-btn")
        page.wait_for_selector("#batch-status:has-text('Completed.')", timeout=240_000)
        page.wait_for_selector("#batch-ccc-wrap .ccc-block", timeout=15_000)

        text = page.inner_text("#batch-ccc-wrap")
        print(text)

        rows = page.eval_on_selector_all(
            "#batch-ccc-wrap .ccc-block:first-child .ccc-table tbody tr", "els => els.length"
        )
        combined_present = "Combined*" in text
        footnote_present = "does not report this figure" in text

        print(f"\nfirst table rows: {rows}")
        print(f"combined row present: {combined_present}")
        print(f"footnote present: {footnote_present}")

        browser.close()

        ok = not errors and combined_present and footnote_present and rows == 3
        print("PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
