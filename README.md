# Adobe-India-Hackathon25

# PDF Structured Outline Extractor — Adobe Hackathon Round 1A Submission

## Project Objective

This project addresses the Round 1A challenge of the Adobe Hackathon: extract a clean, structured document outline from unstructured PDF files. Each PDF may contain complex layouts, multiple heading levels, and inconsistent formatting. Our goal is to automatically generate a machine-readable table of contents from such documents, producing reliable output even in the absence of metadata or manually authored bookmarks.

---

## Key Features

- Heading detection using font size, boldness, and layout positioning
- Accurate H1–H6 outline construction based on visual hierarchy
- Multilingual support across Latin, Indic, and CJK scripts
- Automatic column detection (single, double, and triple column layouts)
- Header and footer detection via repeated content analysis
- Filters out low-density or non-informative lines (e.g., page numbers)
- Clean, valid JSON output with text, heading level, and page number
- Works fully offline, can be extended with OCR as needed

---

## Approach Overview

### 1. Block Extraction

- Used `PyMuPDF`'s `get_text("html")` interface to extract styled text and positional data from each page.
- Cleaned and parsed HTML content to identify individual lines with their bounding boxes, font sizes, and styles.

### 2. Layout Analysis

- Detected headers and footers by comparing repeated lines across even and odd pages.
- Applied clustering to left-aligned coordinates to infer column structure.
- Merged consecutive lines with same font and alignment to reduce fragmentation.

### 3. Font & Typography Scoring

- Computed font statistics across the document to identify dominant and rare font sizes.
- Used z-score thresholds on word count per font to classify fonts into major (heading) and minor (body) classes.
- Tracked changes in font size, bold, and italic status to infer visual hierarchy.

### 4. Heading Assignment

- Identified candidate heading lines from top/zoned regions based on font and alignment.
- Assigned heading levels (H1–H6) by iterating from largest fonts to smallest, from leftmost to rightmost indentation.
- Ensured consistent nesting: once the level increases (e.g., to H3), higher levels (e.g., H1) are no longer reassigned.

---

## Output Format

The output is a valid JSON object:

```json
{
  "title": "Extracted Title or PDF Name",
  "outline": [
    {
      "level": "H1",
      "text": "1. Introduction",
      "page": 0
    },
    {
      "level": "H2",
      "text": "1.1 Background",
      "page": 1
    }
  ]
}
