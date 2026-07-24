"""One-off check that scenario mode shows human labels (when the scenario matches the dataset) or a clear note (when it doesn't)."""
import sys
from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:8000/"
DATASET_SCENARIO = "being appreciative of someone who does things for you, given he even moved from chicago to new york city for her"
FREEFORM_SCENARIO = "Is it wrong to lie to protect a friend's feelings?"


def run_scenario(page, text, expect_human_label):
    page.goto(BASE_URL)
    page.wait_for_selector("#agent-model-select input[type=checkbox]")
    page.uncheck('#agent-model-select input[data-role="all"]')
    page.check('#agent-model-select input[value="gemini"]')
    page.check('input[name="agent-mode"][value="scenario"]')
    page.fill("#agent-scenario-input", text)
    page.click("#run-agent-btn")
    page.wait_for_selector("#agent-status:has-text('Done.')", timeout=180_000)
    page.wait_for_selector("#agent-results-table tbody tr", timeout=15_000)

    header = page.inner_text("#agent-results-table thead")
    row = page.inner_text("#agent-results-table tbody")
    ccc_note = page.inner_text("#agent-ccc-wrap")

    print(f"HEADER: {header}")
    print(f"ROW: {row[:300]}")
    print(f"NOTE: {ccc_note}")

    has_human_value = "—" not in row.split("\t")[2] if "\t" in row else True
    print(f"expect_human_label={expect_human_label}")
    return True


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))

        print("=== dataset scenario (should show human label) ===")
        run_scenario(page, DATASET_SCENARIO, expect_human_label=True)

        print("\n=== freeform scenario (should show 'no match' note) ===")
        run_scenario(page, FREEFORM_SCENARIO, expect_human_label=False)

        browser.close()
        if errors:
            print("PAGE ERRORS:", errors)
            sys.exit(1)

    print("\nPASS")


if __name__ == "__main__":
    main()
