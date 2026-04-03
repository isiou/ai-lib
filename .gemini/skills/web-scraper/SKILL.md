---
name: web-scraper
description: |
  Scrape and extract structured content from any web page URL. Use when:
  - User wants to fetch/scrape/crawl a webpage
  - User needs to extract page metadata (title, description, OG tags, Twitter cards)
  - User wants to analyze a website's content
  - User provides a URL and asks "what's on this page" or similar
  - User needs HTML body content from a URL
---

# Web Scraper

Fetch and parse web page content via the WebPageSnap API.

## API Usage

```bash
curl "https://webpagesnap.com/api/scrape?url=<URL_ENCODED>&format=json"
```

**Parameters:**
- `url`: URL-encoded target webpage (required)
- `format`: Response format, use `json` (required)

## Response Structure

```json
{
  "success": true,
  "url": "https://example.com/",
  "finalUrl": "https://example.com/",
  "format": "json",
  "header": {
    "title": "Page Title",
    "description": "Meta description",
    "keywords": "keyword1, keyword2",
    "author": "Author Name",
    "charset": "utf-8",
    "viewport": "width=device-width, initial-scale=1",
    "ogTitle": "Open Graph Title",
    "ogDescription": "Open Graph Description",
    "ogImage": "https://example.com/og-image.png",
    "ogUrl": "https://example.com",
    "twitterCard": "summary_large_image",
    "twitterTitle": "Twitter Card Title",
    "twitterDescription": "Twitter Card Description",
    "twitterImage": "https://example.com/twitter-image.png"
  },
  "body": "<html>...</html>"
}
```

## Workflow

1. URL-encode the target URL
2. Call the API with curl or WebFetch tool
3. Parse the JSON response
4. Extract relevant data from `header` (metadata) or `body` (HTML content)

## Example

```bash
curl "https://webpagesnap.com/api/scrape?url=https%3A%2F%2Fgithub.com&format=json"
```