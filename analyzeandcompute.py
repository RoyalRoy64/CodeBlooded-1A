import pandas as pd
import numpy as np
from collections import Counter

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

    return dflist  # optional; modifies in-place

def analyze_left_distribution(dflist, round_precision=1):
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
            all_lefts.extend(np.round(df['left'].dropna(), round_precision))

    # Count frequencies
    left_counter = Counter(all_lefts)
    total = sum(left_counter.values())

    # Sort by frequency
    left_stats = sorted(left_counter.items(), key=lambda x: -x[1])
    left_stats_df = pd.DataFrame(left_stats, columns=['left', 'count'])
    left_stats_df['percentage'] = 100 * left_stats_df['count'] / total
    left_stats_df['left'] = left_stats_df['left'].round(round_precision)

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