import pandas as pd
def extract_all_tables_with_titles(page, max_distance=50):
    tables = page.find_tables()
    results = []

    if not tables.tables:
        return results  # empty list

    text_dict = page.get_text("dict")
    blocks = text_dict.get("blocks", [])

    def collect_nearby_text(bbox, direction):
        ref_y = bbox[1] if direction == "above" else bbox[3]
        nearby = []
        for block in blocks:
            if "lines" not in block:
                continue
            block_y = block["bbox"][3] if direction == "above" else block["bbox"][1]
            if direction == "above":
                if block_y < ref_y and (ref_y - block_y) <= max_distance:
                    pass
                else:
                    continue
            elif direction == "below":
                if block_y > ref_y and (block_y - ref_y) <= max_distance:
                    pass
                else:
                    continue
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

    def rank_candidates(candidates):
        sorted_cand = sorted(
            candidates,
            key=lambda x: (x["bold"], x["font_size"] or 0, -abs(x["y"])),
            reverse=True
        )
        return sorted_cand[0] if sorted_cand else None

    for table in tables:
        table_data = table.extract()
        table_df = pd.DataFrame(table_data)
        bbox = table.bbox

        above_texts = collect_nearby_text(bbox, "above")
        below_texts = collect_nearby_text(bbox, "below")

        title = rank_candidates(above_texts) or rank_candidates(below_texts)

        results.append((table_df, title,(above_texts,below_texts)))

    return results
