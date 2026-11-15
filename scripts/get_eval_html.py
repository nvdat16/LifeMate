from playwright.sync_api import sync_playwright

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto('https://challenge.zalo.ai/portal/lifemate', wait_until='networkidle')
        
        # Click Evaluation tab
        for tab in page.locator('div[role="tab"]').all():
            if tab.inner_text().strip() == 'Evaluation':
                tab.click()
                break
        
        page.wait_for_timeout(2000)
        html = page.content()
        with open('docs/portal_full.html', 'w', encoding='utf-8') as f:
            f.write(html)
        print('Saved portal HTML successfully')
        browser.close()

if __name__ == '__main__':
    main()

