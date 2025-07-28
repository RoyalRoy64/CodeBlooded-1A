import pandas as pd
import numpy as np
from collections import Counter
import json
def unify_close_left_values(dflist, tolerance=0.3, min_left=None, max_left=None):

    # 1. Collect all left values from all dfs
    all_lefts = []
    for df in dflist:
        lefts = df['left']
        if min_left is not None:
            lefts = lefts[lefts >= min_left]
        if max_left is not None:
            lefts = lefts[lefts <= max_left]
        all_lefts.extend(lefts.tolist())

    # 2. Sort and cluster the values
    all_lefts = sorted(set(all_lefts))
    clusters = []
    current_cluster = [all_lefts[0]]

    for val in all_lefts[1:]:
        if abs(val - current_cluster[-1]) <= tolerance:
            current_cluster.append(val)
        else:
            clusters.append(current_cluster)
            current_cluster = [val]
    clusters.append(current_cluster)

    # 3. Create a mapping: each value in cluster -> cluster representative (mean or min)
    value_map = {}
    for cluster in clusters:
        rep = round(np.mean(cluster), 3)  # or use min(cluster) or round(np.mean(cluster))
        for val in cluster:
            value_map[val] = rep

    # 4. Apply mapping to all dfs
    for df in dflist:
        df['left'] = df['left'].apply(lambda x: value_map.get(x, x))

    return dflist  

def analyze_left_distribution(dflist):
    """
    Analyzes the distribution of 'left' alignment values across all pages.

    Parameters:
        dflist (list of pd.DataFrame): List of DataFrames, each representing a page.
        round_precision (int): Decimal places to round 'left' values for grouping.

    Returns:
        dict:
            - left_counter (Counter): Frequency of each rounded 'left' value.
            - dominant_lefts (list): Most frequent left positions (likely body alignment).
            - left_stats_df (pd.DataFrame): Summary table with frequency and percentage.
    """
    all_lefts = []

    for page_num, df in enumerate(dflist):
        if 'left' in df.columns and not df.empty:
            valid_lefts = pd.to_numeric(df['left'], errors='coerce').dropna()
            all_lefts.extend(valid_lefts.round(2).tolist())

    # Count frequencies
    left_counter = Counter(all_lefts)
    total = sum(left_counter.values())

    # Sort by frequency
    left_stats = sorted(left_counter.items(), key=lambda x: -x[1])
    left_stats_df = pd.DataFrame(left_stats, columns=['left', 'count'])
    left_stats_df['percentage'] = 100 * left_stats_df['count'] / total
    # Make sure 'left' column is numeric and drop invalid rows
    left_stats_df = left_stats_df[pd.to_numeric(left_stats_df['left'], errors='coerce').notnull()]
    left_stats_df['left'] = left_stats_df['left'].astype(float).round(2)


    # Pick dominant left positions (e.g., >10% of lines)
    dominant_lefts = left_stats_df[left_stats_df['percentage'] > 10]['left'].tolist()

    return {
        'left_counter': left_counter,
        'dominant_lefts': dominant_lefts,
        'left_stats_df': left_stats_df
    }


def analyze_left_clusters(dflist, target_lefts, tolerance=0.1):
    summaries = []
    
    for left_val in target_lefts:
        texts = []
        font_sizes = []
        line_heights = []
        word_counts = []
        bold_flags = []
        italic_flags = []
        sample_texts = []

        for df in dflist:
            sub_df = df[np.isclose(df['left'], left_val, atol=tolerance)]
            if sub_df.empty:
                continue

            texts.extend(sub_df['text'].values)
            font_sizes.extend(sub_df['font_size'].values)
            line_heights.extend(sub_df['line_height'].values)
            word_counts.extend(sub_df['word_count'].values)
            bold_flags.extend(sub_df['bold'].astype(int).values)
            italic_flags.extend(sub_df['italic'].astype(int).values)
            sample_texts.extend(sub_df['text'].head(3).values.tolist())

        if not texts:
            continue

        summaries.append({
            'left': left_val,
            'count': len(texts),
            'avg_font_size': round(np.mean(font_sizes), 2),
            'avg_line_height': round(np.mean(line_heights), 2),
            'avg_word_count': round(np.mean(word_counts), 2),
            'bold_%': round(100 * np.mean(bold_flags), 1),
            'italic_%': round(100 * np.mean(italic_flags), 1),
            'sample_texts': sample_texts
        })

    return pd.DataFrame(summaries)

def analyze_line_spacing(spacing_list, z_thresh_upper=1.2, z_thresh_lower=-0.5):
    """
    Analyzes line spacings and finds the statistically acceptable merging range.

    Parameters:
        spacing_list (list of np.ndarray): Output of compute_line_spacing_per_page
        z_thresh_upper (float): Upper Z-score limit for acceptable spacing
        z_thresh_lower (float): Lower Z-score limit for acceptable spacing

    Returns:
        dict with:
            - mean_spacing (float): Mean spacing
            - std_spacing (float): Standard deviation
            - acceptable_range (tuple): (min_acceptable, max_acceptable) spacing values
            - z_scores (np.ndarray): Z-scores of all spacings
            - all_spacings (np.ndarray): All spacing values concatenated
    """
    all_spacings = np.concatenate(spacing_list)
    
    if len(all_spacings) == 0:
        return {
            'mean_spacing': 0,
            'std_spacing': 0,
            'acceptable_range': (0, 0),
            'z_scores': np.array([]),
            'all_spacings': np.array([])
        }

    mean_spacing = np.mean(all_spacings)
    std_spacing = np.std(all_spacings)

    z_scores = (all_spacings - mean_spacing) / (std_spacing + 1e-6)

    acceptable = all_spacings[
        (z_scores <= z_thresh_upper) & 
        (z_scores >= z_thresh_lower)
    ]
    min_acc, max_acc = (np.min(acceptable), np.max(acceptable)) if len(acceptable) > 0 else (0, 0)

    return {
        'mean_spacing': round(mean_spacing, 2),
        'std_spacing': round(std_spacing, 2),
        'acceptable_range': (round(min_acc, 2), round(max_acc, 2)),
        'z_scores': np.round(z_scores, 2),
        'all_spacings': np.round(all_spacings, 2)
    }


def compute_word_count_stats(dflist):
    """
    Computes average word count per line and per page for a list of DataFrames.
    Returns a summary dict with global average and per-page stats.
    """
    page_stats = []
    total_words = 0
    total_lines = 0

    for i, df in enumerate(dflist):
        # if 'word_count' not in df.columns:
        #     df['word_count'] = df['text'].str.split().apply(len)
        page_word_count = df['word_count'].sum()
        page_line_count = len(df)
        page_avg = page_word_count / page_line_count if page_line_count else 0
        ##########this is not required
        page_stats.append({
            'page': i,
            'lines': page_line_count,
            'words': page_word_count,
            'avg_words_per_line': round(page_avg, 2)
        })

        total_words += page_word_count
        total_lines += page_line_count

    global_avg = total_words / total_lines if total_lines else 0

    return {
        'global_avg_words_per_line': round(global_avg, 2),
        'pages': page_stats
    }

def compute_line_spacing_per_page(dflist):
    """
    Computes the vertical distance (spacing) between subsequent rows 
    based on the 'top' coordinate for each DataFrame in dflist.

    Parameters:
        dflist (list of pd.DataFrame): Each DataFrame represents a page with a 'top' column.

    Returns:
        list of np.ndarray: Each element is an array of vertical distances for that page.
                            If page has <2 rows or no 'top' column, returns empty array.
    """
    spacing_list = []

    for df in dflist:
        if 'top' in df.columns and len(df) >= 2:
            sorted_top = df['top'].sort_values().values
            spacing = np.diff(sorted_top)  # Compute differences between successive 'top' values
            spacing_list.append(spacing)
        else:
            spacing_list.append(np.array([]))  # No spacing info for this page

    return spacing_list

def detect_column_structure_by_clusters(leftstatdf, page_width=612, tolerance=25, debug=False):
    import numpy as np
    import pandas as pd

    def cluster_lefts(df, tolerance):
        sorted_df = df.sort_values('left').reset_index(drop=True)
        clusters = []
        current = [sorted_df.loc[0, 'left']]
        current_sum = sorted_df.loc[0, 'count']

        for i in range(1, len(sorted_df)):
            val = sorted_df.loc[i, 'left']
            count = sorted_df.loc[i, 'count']
            if abs(val - np.mean(current)) <= tolerance:
                current.append(val)
                current_sum += count
            else:
                clusters.append({'center': np.mean(current), 'count': current_sum})
                current = [val]
                current_sum = count
        clusters.append({'center': np.mean(current), 'count': current_sum})
        return pd.DataFrame(clusters)

    clusters = cluster_lefts(leftstatdf, tolerance)

    # Assign zones based on center
    def zone(center):
        if center < 0.33 * page_width:
            return 'left'
        elif center < 0.66 * page_width:
            return 'middle'
        else:
            return 'right'

    clusters['zone'] = clusters['center'].apply(zone)

    zone_counts = clusters.groupby('zone')['count'].sum()
    total_lines = leftstatdf['count'].sum()
    strong_zones = zone_counts[zone_counts >= 0.20 * total_lines].index.tolist()

    sorted_clusters = clusters.sort_values('count', ascending=False).reset_index(drop=True)
    if len(sorted_clusters) >= 2:
        top1, top2 = sorted_clusters.iloc[0], sorted_clusters.iloc[1]
        horizontal_gap = abs(top1['center'] - top2['center'])

        if horizontal_gap >= 0.3 * page_width and top1['count'] >= 0.2 * total_lines and top2['count'] >= 0.2 * total_lines:
            return 2

    if all(z in strong_zones for z in ['left', 'middle', 'right']):
        return 3
    elif len(strong_zones) >= 2:
        return 2
    else:
        return 1


def enrich_masterdf_with_heading_signals(masterdf, num_cols=1, page_width=612.0):
    """
    Enriches master DataFrame with heading-related features based on layout, font differences,
    and style switches.

    Parameters:
        masterdf (pd.DataFrame): DataFrame with parsed PDF content.
        num_cols (int): Number of text columns in layout (1, 2, or 3).
        page_width (float): Width of the page in points (default is 612.0 for A4).

    Returns:
        pd.DataFrame: The enriched DataFrame with new signal columns.
    """
    import numpy as np
    df = masterdf.copy()

    # Step 1: Assign left_zone based on number of columns
    if num_cols == 1:
        df['left_zone'] = 0
    elif num_cols == 2:
        midpoint = page_width / 2
        df['left_zone'] = df['left'].apply(lambda x: 0 if x < midpoint else 1)
    elif num_cols == 3:
        col1_end = page_width / 3
        col2_end = 2 * page_width / 3
        df['left_zone'] = df['left'].apply(
            lambda x: 0 if x < col1_end else (1 if x < col2_end else 2)
        )
    else:
        raise ValueError("Invalid number of columns. Must be 1, 2, or 3.")

    # Step 2: Assign left_level within each zone based on heading-like candidates
    df['left_level'] = -1
    candidate_filter = df['word_count'] > 0

    for zone in sorted(df['left_zone'].unique()):
        zone_candidates = df[(df['left_zone'] == zone) & candidate_filter]
        unique_left_sorted = sorted(zone_candidates['left'].unique())  # More left = higher level
        left_to_level = {val: i for i, val in enumerate(unique_left_sorted)}
        df.loc[df['left_zone'] == zone, 'left_level'] = df[df['left_zone'] == zone]['left'].map(left_to_level)

    # Step 3: Font size change with next line (positive = current is larger)
    df['font_size_change'] = df['font_size'] - df['font_size'].shift(-1)
    df.iloc[-1]['font_size_change'] = 0
    # Step 4: Font family change with next line
    df['font_family_change'] = (df['font_family'] != df['font_family'].shift(-1)).astype(int)

    # Step 5: Bold/Italic style switch detection
    df['bold_change'] = (
        df['bold'].astype(int) - df['bold'].shift(-1).fillna(False).astype(int)
    )
    df['italic_change'] = (
        df['italic'].astype(int) - df['italic'].shift(-1).fillna(False).astype(int)
    )
    df.loc[df.index[-1], ['font_size_change', 'font_family_change', 'bold_change', 'italic_change']] = 0

    return df.reset_index(drop=True)

def extract_zoned_rows_from_masterdf(
    master_df,
    removed_pages=None,
    major_fonts=None,
    num_cols=1,
    page_width=612.0,
    page_height=792.0
):
    """
    Extracts candidate lines from top & left zones depending on layout,
    and also adds relevant lines from non-header/footer pages if needed.

    Parameters:
        master_df (pd.DataFrame): Full document dataframe.
        removed_pages (list or None): Pages with removed headers/footers.
        major_fonts (list or set): Font sizes considered important.
        num_cols (int): Number of columns in layout.
        page_width (float): Page width in points.
        page_height (float): Page height in points.

    Returns:
        pd.DataFrame: Filtered DataFrame of candidate heading lines.
    """
    df = master_df.copy()
    df['left'] = df['left'].round(2)

    total_pages = df['page_number'].max() + 1

    if removed_pages is None:
        removed_pages = list(range(total_pages))
        process_all_pages = True
    else:
        process_all_pages = False

    # Define masks for top and left zones
    top_mask = df['top'] < page_height * 0.2
    if num_cols == 1:
        left_mask = df['left'] < page_width * 0.2
    elif num_cols == 2:
        left_mask = ((df['left'] < page_width * 0.15) |
                     ((df['left'] > page_width * 0.5) & (df['left'] < page_width * 0.65)))
    elif num_cols == 3:
        left_mask = ((df['left'] < page_width * 0.07) |
                     ((df['left'] > page_width * 0.33) & (df['left'] < page_width * 0.4)) |
                     ((df['left'] > page_width * 0.66) & (df['left'] < page_width * 0.73)))
    else:
        raise ValueError("Invalid number of columns")

    # Step 1: Process removed or all pages
    if process_all_pages:
        primary_df = df[top_mask & left_mask]
    else:
        primary_df = df[df['page_number'].isin(removed_pages) & top_mask & left_mask]

    # Step 2: Include strong candidates from non-removed pages
    if not process_all_pages and major_fonts is not None:
        non_removed_mask = ~df['page_number'].isin(removed_pages)
        secondary_df = df[non_removed_mask & ((df['parabool'] == False) | (df['font_size'].isin(major_fonts)))]
        final_df = pd.concat([primary_df, secondary_df], ignore_index=False)
    else:
        final_df = primary_df

    return final_df



def assign_heading_levels(top_candidates):
    """
    Assign hierarchical heading levels (H1, H2, ...) to rows in top_candidates,
    based on font_size (desc), left_level (asc), and bold status.
    
    Rules:
    - Start from H1 and move down (H2, H3...); never go back up.
    - Within each font size, process more-left items before more-right.
    - Bold rows get priority. If no bold, non-bold gets the next available level.
    
    Returns:
        DataFrame with new column `heading_level` (e.g., 'H1', 'H2', ...)
    """
    df = top_candidates.copy()
    df = df.sort_values(by=['font_size', 'left_level'], ascending=[False, True]).reset_index()

    # Initialize heading_level column
    df['heading_level'] = None

    used_indices = set()
    current_level = 1

    sorted_fonts = sorted(df['font_size'].unique(), reverse=True)
    
    for font in sorted_fonts:
        font_group = df[df['font_size'] == font]
        sorted_lefts = sorted(font_group['left_level'].unique())

        for left in sorted_lefts:
            sub_group = font_group[font_group['left_level'] == left]

            # Bold rows first
            bold_rows = sub_group[(sub_group['bold'] == True) & (~sub_group['index'].isin(used_indices))]
            if not bold_rows.empty:
                df.loc[df['index'].isin(bold_rows['index']), 'heading_level'] = f'H{current_level}'
                used_indices.update(bold_rows['index'])
                current_level += 1  # Drop level for next group

            # Non-bold rows
            non_bold_rows = sub_group[(sub_group['bold'] == False) & (~sub_group['index'].isin(used_indices))]
            if not non_bold_rows.empty:
                df.loc[df['index'].isin(non_bold_rows['index']), 'heading_level'] = f'H{current_level}'
                used_indices.update(non_bold_rows['index'])
                current_level += 1

    # Drop temp index and return
    return df.drop(columns=['index'])

def write_outline_json(df, output_dir, pdf_file, title=""):
    """
    Writes outline JSON from the dataframe.
    If df is None or empty, creates an empty outline structure.
    Assumes `heading_level`, `text`, and `page_number` are in df.
    """
    if df is None or df.empty:
        outline_data = {
            "title": title,
            "outline": []
        }
    else:
        outline = []
        for _, row in df.iterrows():
            outline.append({
                "level": row["heading_level"],  # Already formatted as 'H1', 'H2', etc.
                "text": str(row["text"]),
                "page": int(row["page_number"])
            })

        outline_data = {
            "title": title,
            "outline": outline
        }

    # Save to file
    output_file = output_dir / f"{pdf_file.stem}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(outline_data, f, indent=2, ensure_ascii=False)

    print(f"Processed {pdf_file.name} -> {output_file.name}")
