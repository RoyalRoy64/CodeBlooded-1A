
# PDF Structured Outline Extractor — Adobe Hackathon 2024 (Round 1A Submission)

## Overview

This project addresses the Round 1A challenge of the Adobe India Hackathon: extracting a clean, hierarchical document outline from unstructured PDF files. The solution is designed to handle complex layouts, multi-column formatting, multilingual content, and varying heading styles — generating a machine-readable table of contents (TOC) in JSON format.

---

## Objectives

- Automatically identify and classify headings (H1–H6) from PDF documents
- Handle variations in layout, fonts, and structure without relying on metadata
- Support multilingual content and visual hierarchy inference
- Operate fully offline and within strict performance constraints

---

## Key Features

- Heading detection using **font size**, **boldness**, and **layout positioning**
- Accurate **H1–H6 outline construction** based on visual hierarchy
- **Multilingual support**: Handles Latin, Indic, and CJK scripts
- **Automatic column detection** for single, double, and triple column layouts
- **Header and footer removal** using repeated line analysis
- Filters out **low-density/non-informative lines** (e.g., page numbers)
- Outputs **clean, valid JSON** with heading level, text, and page number
- Offline-compatible; OCR support (via Tesseract) can be optionally integrated

---

## Technical Approach

### 1. Block Extraction
- Utilized `PyMuPDF`’s `get_text("html")` to extract styled text and bounding boxes
- Parsed HTML content into lines with positional metadata and font styles

### 2. Layout Analysis
- Identified headers and footers via repeated content across pages
- Clustered lines using `x0` coordinates to infer column structure
- Merged consecutive lines with similar font and alignment for cohesion

### 3. Font & Typography Scoring
- Analyzed font distribution using word frequency and z-scores
- Classified fonts into heading vs body based on rarity and prominence
- Tracked boldness and italicization for additional context

### 4. Heading Level Assignment
- Selected heading candidates from key zones (top, margin, center)
- Assigned levels (H1–H6) in descending order of font size and indentation
- Enforced structural nesting to preserve outline hierarchy

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
````

---

## How to Run (Docker)

### Build the image

```bash
docker build --platform linux/amd64 -t pdf-outline-extractor .
```

### Run the container

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  pdf-outline-extractor
```

* Place PDF files inside the `/input` directory.
* JSON outputs will be saved in `/output` with matching filenames.

---

## Project Structure

```
.
├── app/
│   ├── extractor.py         # Core extraction logic
│   ├── layout_utils.py      # Column detection, font scoring
│   ├── ocr_fallback.py      # Tesseract-based OCR logic (optional)
│   └── main.py              # Entry point
├── input/                   # Input PDFs go here
├── output/                  # JSON outputs saved here
├── Dockerfile
└── README.md
```

---

## Testing

The system has been tested on:

* Text-based PDFs with varying structure
* Multi-column scientific papers
* Hindi, Japanese, Korean, and Russian documents (with OCR fallback)
* Scanned pages and low-font-embedding documents

Execution time: consistently under 10 seconds for 50-page PDFs.

---

## Bonus Highlights

* Multilingual handling with support for `eng`, `hin`, `jpn`, `kor`, `rus`
* Works offline with optional OCR fallback
* Robust to layout variance, font inconsistencies, and content density
* Scalable to large batch processing via Docker

---

## Authors

* Spandan Roy
* Honey Priya

---

*Built for Adobe’s vision of smarter document intelligence. Not just parsing — understanding.*

```

