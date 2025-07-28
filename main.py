import pymupdf
# import pymupdf4llm
from pathlib import Path
import pandas as pd
import numpy as np
import re
import html
import json

from collections import Counter
from extractandclean    import extract_blocks_and_distributions , extract_heading_summary , extract_crucial_pattern_lines , filter_and_clean_gibberish , decode_text_in_dflist_to_utf8
from merge              import merge_consecutive_same_font_and_left , combine_dflist_to_master_df
from headerfooter       import find_alternate_page_repeats_split , clean_pages_of_even_odd_repeats
from tables             import extract_all_tables_with_titles
from fontstat           import generate_font_stats , get_global_font_counter , get_rare_large_fonts , map_fonts_to_heading_levels , get_significant_large_fonts , create_font_level_map , get_top_fonts , classify_font_size
from analyzeandcompute  import unify_close_left_values , analyze_left_distribution , analyze_left_clusters , analyze_line_spacing , compute_word_count_stats , compute_line_spacing_per_page , detect_column_structure_by_clusters , enrich_masterdf_with_heading_signals  , extract_zoned_rows_from_masterdf , write_outline_json ,assign_heading_levels

#workflow
"""
    1. Input file (string)
    2. Extract Blocks to List of DF
    3. Unify Close Left Values
    4. Find and Clean Headers and Footers
    5. Extract Tables
    6. Extract crucial pattern lines
    7. Merge consecutive same font and left
    8. Font Stats , left distribution , line spacing
"""
#tesseract ocr
#pip install pymupdf numpy pandas


input_dir = Path("/app/input")
output_dir = Path("/app/output")

output_dir.mkdir(parents=True, exist_ok=True)
pdf_files = list(input_dir.glob("*.pdf"))

for pdf_file in pdf_files:
    try:
        doc = pymupdf.open(pdf_file)
        totalpage = doc.page_count

        # Handle empty PDF case
        if totalpage == 0:
            write_outline_json(None, output_dir, pdf_file, title="")
            print(f"Skipped empty PDF: {pdf_file.name}")
            continue  # <== Skip further processing for this file

        page = doc[0]
        width, height = page.rect.width, page.rect.height
        doctoc = doc.get_toc()
        docmetadata = doc.metadata
        dflist = []
        mid = totalpage // 2
        for i in range(totalpage):
            page = doc.load_page(i)
            html = page.get_text("html")
            blocks= extract_blocks_and_distributions(html)
            dflist.append(pd.DataFrame(blocks))
        dflist = decode_text_in_dflist_to_utf8(dflist)
        dflist = unify_close_left_values(dflist, tolerance=width*0.01, min_left=0, max_left=width)


        # for df in dflist:
        #     if not df.empty:
        dflist   =   [filter_and_clean_gibberish(df, min_alnum_ratio=0.15, min_length=1) for df in dflist] 

        dfcrucial = extract_crucial_pattern_lines(dflist)

        even_df, odd_df =   find_alternate_page_repeats_split(dflist, mid=totalpage // 2)
        dflist, removed_pages = clean_pages_of_even_odd_repeats(dflist, even_df, odd_df)
        print("Removed header/footer from pages:", removed_pages)
        
        average_words_per_line_per_column    =   compute_word_count_stats(dflist)['global_avg_words_per_line']
        font_stats_df = generate_font_stats(dflist)
        fontsize_word_counts = pd.Series(dtype=float)  # empty default

        if (
            not font_stats_df.empty and
            'font_size' in font_stats_df.columns and
            'total_words' in font_stats_df.columns
        ):
            fontsize_word_counts = (
                font_stats_df.groupby('font_size')['total_words']
                .sum()
                .sort_values(ascending=False)
            )
            major_fonts = get_significant_large_fonts(fontsize_word_counts, z_thresh=0.55)['large_fonts']
            minor_fonts = get_significant_large_fonts(fontsize_word_counts, z_thresh=-0.45)['large_fonts']
        else:
            major_fonts = []
            minor_fonts = []

        mean_lineheight = analyze_line_spacing(compute_line_spacing_per_page(dflist))['mean_spacing']
        merged_dflist = merge_consecutive_same_font_and_left(dflist,list(fontsize_word_counts.keys()),[mean_lineheight*0.4,mean_lineheight*1.75])

        leftdf = analyze_left_distribution(dflist)['left_stats_df'] # 'left_counter': left_counter,# 'dominant_lefts': dominant_lefts,# 'left_stats_df': left_stats_df
        leftdf.loc[(leftdf.left<=204) & (leftdf.percentage>=1)]
        
        notmasterdf = combine_dflist_to_master_df(dflist)
        master_df = combine_dflist_to_master_df(merged_dflist)
        title = docmetadata['title']
        if title == '':
            # Fallback 1: Use first page with most words
            page_word_counts = [(i, len(df['text'])) for i, df in enumerate(dflist) if not df.empty and 'text' in df.columns]
            if page_word_counts:
                # Get the page with the maximum words
                best_page_index = max(page_word_counts, key=lambda x: x[1])[0]
                df_top = dflist[best_page_index]

                # Get top 3 fonts (by count)
                if 'font_family' in df_top.columns:
                    top_fonts = df_top['font_family'].value_counts().head(3).index.tolist()
                    title = ", ".join(top_fonts)
                else:
                    title = pdf_file.stem
            else:
                # Fallback 2: Use filename
                title = pdf_file.stem
        if master_df.empty or notmasterdf.empty:
            write_outline_json(None, output_dir, pdf_file, title=title) 
        else:
            master_df['left'] = master_df['left'].round(2)
            notmasterdf['left'] = notmasterdf['left'].round(2)
            leftstatdf = analyze_left_distribution([notmasterdf])['left_stats_df'] # 'left_counter': left_counter,# 'dominant_lefts': dominant_lefts,# 'left_stats_df': left_stats_df

            wordandfontsize_limit_masterdf = (master_df.loc[master_df.word_count<11])
            unique_left_sorted = np.array(sorted(set(val for val in wordandfontsize_limit_masterdf['left'].unique())))
            num_cols = detect_column_structure_by_clusters(leftstatdf, page_width=width, tolerance=25)
            enricheddf = enrich_masterdf_with_heading_signals(master_df, num_cols, page_width=width)
            top_candidates = extract_zoned_rows_from_masterdf(
                master_df=master_df,
                removed_pages=removed_pages,     # or None
                major_fonts=major_fonts,         # precomputed list/set of sizes
                num_cols=num_cols,
                page_width=width,
                page_height=height
            )
            enriched_top_candidates = assign_heading_levels(top_candidates)
            write_outline_json(enriched_top_candidates, output_dir, pdf_file, title=title)
            print(f"Finished processing {pdf_file.name} — {title}")

    except Exception as e:
        print(f"Error processing {pdf_file.name}: {e}")