"""One-off check that the CCC tables render in the UI after a batch job and after an agent run."""
import sys
from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:8000/"


def test_batch(page):
    page.goto(BASE_URL)
    page.wait_for_selector("#batch-model-select input[type=checkbox]")
    page.uncheck('#batch-model-select input[data-role="all"]')
    page.check('#batch-model-select input[value="gemini"]')
    page.check('#batch-model-select input[value="lama"]')
    page.fill("#sample-size-input", "3")
    page.click("#run-batch-btn")
    page.wait_for_selector("#batch-status:has-text('Completed.')", timeout=180_000)
    page.wait_for_selector("#batch-ccc-wrap .ccc-block", timeout=15_000)
    tables = page.eval_on_selector_all("#batch-ccc-wrap .ccc-block", "els => els.length")
    text = page.inner_text("#batch-ccc-wrap")
    print("[batch] ccc blocks:", tables)
    print("[batch] ccc text preview:\n", text[:600])
    return tables >= 1


def test_agent(page):
    page.goto(BASE_URL)
    page.wait_for_selector("#agent-model-select input[type=checkbox]")
    page.uncheck('#agent-model-select input[data-role="all"]')
    page.check('#agent-model-select input[value="gemini"]')
    page.check('input[name="agent-mode"][value="random"]')
    page.fill("#agent-sample-size-input", "2")
    page.click("#run-agent-btn")
    page.wait_for_selector("#agent-status:has-text('Done.')", timeout=180_000)
    page.wait_for_selector("#agent-ccc-wrap .ccc-block", timeout=15_000)
    tables = page.eval_on_selector_all("#agent-ccc-wrap .ccc-block", "els => els.length")
    text = page.inner_text("#agent-ccc-wrap")
    print("[agent] ccc blocks:", tables)
    print("[agent] ccc text preview:\n", text[:600])
    return tables >= 1


def main():
    ok = True
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))

        ok &= test_batch(page)
        ok &= test_agent(page)

        browser.close()
        if errors:
            print("PAGE ERRORS:", errors)
            ok = False

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
