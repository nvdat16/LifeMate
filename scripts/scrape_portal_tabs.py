import time
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

USER_DATA_DIR = Path(str(Path(__file__).resolve().parent / ".browser_profile"))
URL = "https://challenge.zalo.ai/portal/lifemate"

def main():
    print(f"Connecting to {URL}...")
    with sync_playwright() as p:
        # Check if user data dir exists
        if USER_DATA_DIR.exists():
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(USER_DATA_DIR),
                headless=True,
                viewport={"width": 1600, "height": 1000}
            )
            page = context.pages[0] if context.pages else context.new_page()
        else:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            context = browser

        page.goto(URL, wait_until="networkidle", timeout=30000)
        time.sleep(3)

        print("Page Title:", page.title())

        # Inspect all possible tabs inside the portal
        # Specifically targeting: Overview, Description, Evaluation, Data, Rule, Submission Guidance
        target_tab_names = [
            "Overview", "Description", "Evaluation", "Data", "Rule", "Submission Guidance", "Rules", "Submission"
        ]

        scraped_data = {}

        # 1. Capture current view first
        body_text = page.inner_text("body")
        scraped_data["initial_view"] = body_text

        # 2. Look for elements with text matching target_tab_names
        print("\nScanning for tabs...")
        for tab_name in target_tab_names:
            print(f"--- Checking tab: {tab_name} ---")
            # Try to find and click element with exact or close text
            locators = [
                page.locator(f"text='{tab_name}'"),
                page.locator(f"button:has-text('{tab_name}')"),
                page.locator(f"div[role='tab']:has-text('{tab_name}')"),
                page.locator(f"a:has-text('{tab_name}')"),
            ]
            
            clicked = False
            for loc in locators:
                count = loc.count()
                for i in range(count):
                    el = loc.nth(i)
                    try:
                        if el.is_visible():
                            txt = el.inner_text().strip()
                            # Check if it matches closely
                            if txt.lower() == tab_name.lower():
                                print(f"Found clickable element for '{tab_name}' (match exact: {txt})")
                                el.click()
                                time.sleep(2)
                                page.wait_for_load_state("networkidle", timeout=5000)
                                content = page.inner_text("body")
                                scraped_data[tab_name] = content
                                print(f"Extracted {len(content)} characters for '{tab_name}'")
                                clicked = True
                                break
                    except Exception as e:
                        pass
                if clicked:
                    break
            
            if not clicked:
                print(f"Tab '{tab_name}' not found or not visible.")

        # Save all scraped contents to a JSON file for deep analysis
        out_json = Path(__file__).resolve().parents[1] / "docs" / "portal_raw_tabs.json"
        out_json.write_text(json.dumps(scraped_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved raw data to {out_json}")

        context.close()

if __name__ == "__main__":
    main()

