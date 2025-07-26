
def extract_table_with_title(page, max_distance=50):
    """
    Extracts the first table and possible title text (above or below),
    along with formatting metadata (font size, bold, italic, underline).

    Parameters:
        page (fitz.Page): The PDF page object.
        max_distance (float): Max vertical distance in points to search above or below the table.

    Returns:
        tuple:
            - pd.DataFrame: The extracted table.
            - dict: Info about possible title (or footer), with formatting.
    """
    tables = page.find_tables()
    print(f"{len(tables.tables)} table(s) found.")

    if not tables.tables:
        return None, None

    table = tables[0]
    table_data = table.extract()
    table_df = pd.DataFrame(table_data)
    print("First table extracted:")
    pprint(table_data)

    bbox = table.bbox  # (x0, y0, x1, y1)
    text_dict = page.get_text("dict")

    def collect_nearby_text(blocks, ref_y, direction="above"):
        nearby = []
        for block in blocks:
            if "lines" not in block:
                continue
            block_y = block["bbox"][3] if direction == "above" else block["bbox"][1]
            is_above = direction == "above" and block_y < ref_y and (ref_y - block_y) <= max_distance
            is_below = direction == "below" and block_y > ref_y and (block_y - ref_y) <= max_distance
            if is_above or is_below:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            nearby.append({
                                "y": block_y,
                                "text": text,
                                "font_size": span.get("size"),
                                "bold": "bold" in span.get("font", "").lower(),
                                "italic": "italic" in span.get("font", "").lower(),
                                "underline": span.get("flags", 0) & 4 != 0
                            })
        return nearby

    above_texts = collect_nearby_text(text_dict.get("blocks", []), ref_y=bbox[1], direction="above")
    below_texts = collect_nearby_text(text_dict.get("blocks", []), ref_y=bbox[3], direction="below")

    def rank_candidates(candidates):
        # Prioritize bold or larger font entries
        sorted_cand = sorted(
            candidates,
            key=lambda x: (x["bold"], x["font_size"] or 0, -abs(x["y"])),  # bold > font size > closest
            reverse=True
        )
        return sorted_cand[0] if sorted_cand else None

    title_above = rank_candidates(above_texts)
    title_below = rank_candidates(below_texts)

    # Choose the better one by some priority (e.g. favor above)
    chosen_title = title_above or title_below

    if chosen_title:
        print("Possible table label:")
        pprint(chosen_title)
    else:
        print("No meaningful title/label found near the table.")

    return table_df, chosen_title