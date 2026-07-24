"""
One-off UI smoke test driving the real web frontend with a real browser
(via Playwright, using the system's installed Chrome). Not part of the
app; just used to verify the agent panel wiring end-to-end.

Requires: pip install playwright (and a Chrome install, used via channel="chrome").
"""
import sys

from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:8000/"


def run_mode(page, mode: str, page_errors: list):
    page.goto(BASE_URL)
    page.wait_for_selector("#agent-model-select input[type=checkbox]")

    page.uncheck('#agent-model-select input[data-role="all"]')
    page.check('#agent-model-select input[value="gemini"]')

    if mode == "scenario":
        page.check('input[name="agent-mode"][value="scenario"]')
        page.fill("#agent-scenario-input", "Is it wrong to lie to protect a friend's feelings?")
    elif mode == "random":
        page.check('input[name="agent-mode"][value="random"]')
        page.fill("#agent-sample-size-input", "2")

    page.click("#run-agent-btn")

    # Prove the log is live, not just populated at the end: it must appear
    # (and have at least one line) WHILE the agent is still running.
    live_log_seen = False
    try:
        page.wait_for_selector("#agent-log-wrap:not([hidden])", timeout=60_000)
        mid_run_lines = page.eval_on_selector_all("#agent-log .agent-log-line", "els => els.length")
        live_log_seen = mid_run_lines > 0
        print(f"[{mode}] LOG LINES WHILE RUNNING: {mid_run_lines}")
    except Exception as e:
        print(f"[{mode}] (log did not appear before completion: {e})")

    page.wait_for_selector("#agent-status:has-text('Done.')", timeout=180_000)

    report_html = page.inner_html("#agent-report")
    status_text = page.inner_text("#agent-status")
    table_rows = page.eval_on_selector_all("#agent-results-table tbody tr", "els => els.length")
    csv_btn_disabled = page.eval_on_selector("#download-agent-csv-btn", "el => el.disabled")
    final_log_lines = page.eval_on_selector_all("#agent-log .agent-log-line", "els => els.length")
    log_text = page.inner_text("#agent-log")

    print(f"[{mode}] STATUS: {status_text}")
    print(f"[{mode}] REPORT HTML LENGTH: {len(report_html)}")
    print(f"[{mode}] TABLE ROWS: {table_rows}")
    print(f"[{mode}] CSV BUTTON ENABLED: {not csv_btn_disabled}")
    print(f"[{mode}] FINAL LOG LINES: {final_log_lines}")
    print(f"[{mode}] LOG PREVIEW:\n{log_text[:500]}")

    if page_errors:
        print(f"[{mode}] FAIL: page errors present: {page_errors}")
        return False
    if "Done." not in status_text:
        print(f"[{mode}] FAIL: status did not reach Done.")
        return False
    if mode == "random" and table_rows == 0:
        print(f"[{mode}] FAIL: expected comparison table rows for random-sample mode")
        return False
    if final_log_lines == 0:
        print(f"[{mode}] FAIL: expected agent activity log lines")
        return False
    if not live_log_seen:
        print(f"[{mode}] FAIL: log did not appear live during the run")
        return False
    print(f"[{mode}] PASS")
    return True


def main():
    modes = sys.argv[1:] or ["scenario", "random"]
    all_ok = True

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)

        for mode in modes:
            page_errors = []
            page = browser.new_page()
            page.on("pageerror", lambda exc: page_errors.append(str(exc)))
            all_ok &= run_mode(page, mode, page_errors)
            page.close()

        browser.close()

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
