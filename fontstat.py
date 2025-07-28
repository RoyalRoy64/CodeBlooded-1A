import pandas as pd
import numpy as np
from collections import Counter

def generate_font_stats(dflist):
    all_stats = []

    for i, df in enumerate(dflist):
        if df.empty or 'font_size' not in df.columns:
            continue
        # Group by font size and family
        grouped = df.groupby(['font_size', 'font_family'])

        for (font_size, font_family), group in grouped:
            total_rows = len(group)
            total_words = group['word_count'].sum() if 'word_count' in group.columns else 0
            avg_word_density = group['word_density'].mean() if 'word_density' in group.columns else 0

            all_stats.append({
                'page_number': i,
                'font_size': font_size,
                'font_family': font_family,
                'count': total_rows,
                'total_words': total_words,
                'avg_word_density': round(avg_word_density, 2)
            })

    return pd.DataFrame(all_stats)

def get_global_font_counter(font_stats_df):
    """
    Aggregates font counts across all pages from font_stats_df
    to recreate a usable font_counter object.
    """
    if font_stats_df.empty:
        return Counter()
    return Counter(dict(
        font_stats_df.groupby('font_size')['count'].sum()
    ))


def get_rare_large_fonts(font_counter, method='rms', freq_filter='average'):
    """
    Identifies large fonts (likely headings) that are:
      - larger than the body font size
      - used less frequently than a frequency threshold

    Parameters:
        font_counter (Counter): font_size -> count
        method (str): 'rms' or 'average' to compute body font size
        freq_filter (str|float): 'average', 'median', or numeric threshold

    Returns:
        body_font (float): Computed body font size
        rare_large_fonts (list): Sorted font sizes likely used for headings
    """
    font_sizes = np.array(list(font_counter.keys()), dtype=float)
    counts = np.array([int(font_counter[fs]) for fs in font_sizes])

    # Step 1: Compute the body font
    if method == 'rms':
        body_font = np.sqrt(np.sum((font_sizes ** 2) * counts) / np.sum(counts))
    else:  # 'average'
        body_font = np.sum(font_sizes * counts) / np.sum(counts)

    # Step 2: Determine frequency threshold
    if freq_filter == 'average':
        freq_threshold = np.mean(counts)
    elif freq_filter == 'median':
        freq_threshold = np.median(counts)
    elif isinstance(freq_filter, (int, float)):
        freq_threshold = freq_filter
    else:
        raise ValueError("Invalid freq_filter. Use 'average', 'median', or a numeric value.")

    # Step 3: Filter large, rare fonts
    rare_large_fonts = [
        float(fs) for fs in font_counter
        if fs > body_font and font_counter[fs] < freq_threshold
    ]

    return round(body_font, 2), sorted(rare_large_fonts, reverse=True)

def map_fonts_to_heading_levels(rare_fonts):
    """
    Maps a list of rare large fonts to H1, H2, H3 levels.

    Logic:
    - 1 → H1
    - 2 → H1, H2
    - 3 → H1, H2, H3
    - 4 → H1, H2, H3, H3
    - 5 → H1, H2, H2, H3, H3
    - 6 → H1, H1, H2, H2, H3, H3
    - 7+ → Only first 6 fonts are used as per 6-rule

    Parameters:
        rare_fonts (list[float]): Sorted list of large rare font sizes (descending)

    Returns:
        dict: Mapping {font_size: heading_level}
    """
    rare_fonts = sorted(rare_fonts, reverse=True)[:6]  # cap at 6 fonts
    n = len(rare_fonts)
    heading_map = {}

    if n == 1:
        heading_levels = ['H1']
    elif n == 2:
        heading_levels = ['H1', 'H2']
    elif n == 3:
        heading_levels = ['H1', 'H2', 'H3']
    elif n == 4:
        heading_levels = ['H1', 'H2', 'H3', 'H3']
    elif n == 5:
        heading_levels = ['H1', 'H2', 'H2', 'H3', 'H3']
    else:  # n == 6
        heading_levels = ['H1', 'H1', 'H2', 'H2', 'H3', 'H3']

    for fs, level in zip(rare_fonts, heading_levels):
        heading_map[fs] = level

    return heading_map

def create_font_level_map(font_counter):
    """Maps largest font sizes to H1, H2, H3; everything else is Body."""
    sorted_fonts = sorted(font_counter.keys(), reverse=True)
    return {
        sorted_fonts[0]: "H1" if len(sorted_fonts) > 0 else None,
        sorted_fonts[1]: "H2" if len(sorted_fonts) > 1 else None,
        sorted_fonts[2]: "H3" if len(sorted_fonts) > 2 else None
    }


def get_significant_large_fonts(font_word_counts, z_thresh=0.5, small_z_thresh=-0.5):
    """
    Statistically selects font sizes significantly larger than the body font.
    Also returns the most frequent body font(s) and common small fonts.

    Parameters:
        font_word_counts (pd.Series): Series with index as font_size and values as total word count.
        z_thresh (float): Z-score threshold to classify a font as significantly larger than the body font.
        small_z_thresh (float): Z-score threshold to classify small fonts (e.g., footnotes).

    Returns:
        dict:
            - body_font (float): Estimated body font size (weighted mean).
            - most_common_body_fonts (list): Font sizes closest to the mean with high frequency.
            - large_fonts (list): Font sizes considered larger than body font.
            - small_fonts (list): Font sizes significantly smaller than body font.
            - z_scores (pd.Series): Z-scores of all fonts.
    """
    font_sizes = font_word_counts.index.to_numpy(dtype=float)
    word_counts = font_word_counts.values.astype(float)

    # Step 1: Weighted body font calculation
    weighted_mean = np.sum(font_sizes * word_counts) / np.sum(word_counts)
    weighted_var = np.sum(((font_sizes - weighted_mean) ** 2) * word_counts) / np.sum(word_counts)
    weighted_std = np.sqrt(weighted_var)

    # Step 2: Z-score calculation
    z_scores = (font_sizes - weighted_mean) / (weighted_std + 1e-6)  # numerical safety

    # Step 3: Classify fonts
    large_fonts = font_sizes[z_scores > z_thresh]
    small_fonts = font_sizes[z_scores < small_z_thresh]

    # Step 4: Find body font(s) close to mean and frequent
    body_range = 0.25  # tolerance in font size units
    body_font_candidates = {
        fs: wc for fs, wc in font_word_counts.items()
        if abs(fs - weighted_mean) <= body_range
    }
    sorted_body_fonts = sorted(body_font_candidates.items(), key=lambda x: -x[1])
    most_common_body_fonts = [round(fs, 2) for fs, _ in sorted_body_fonts[:2]]

    return {
        'body_font': round(weighted_mean, 2),
        'most_common_body_fonts': most_common_body_fonts,
        'large_fonts': sorted(large_fonts, reverse=True),
        'small_fonts': sorted(small_fonts),
        'z_scores': pd.Series(z_scores, index=font_sizes).round(2)
    }


def get_top_fonts(font_counter, top_n=3):
    """Returns the top_n most frequent font sizes in descending order."""
    font_freq = font_counter.most_common(top_n)
    return [size for size, _ in font_freq]
def classify_font_size(size, font_to_level):
    """Returns level for a given font size using pre-defined mapping."""
    return font_to_level.get(size, "Body")
def label_font_levels(cleaned_dflist, font_to_level):
    """Adds a 'level' column to each cleaned page's DataFrame."""
    for i, df in enumerate(cleaned_dflist):
        if 'font_size' not in df.columns:
            continue
        cleaned_dflist[i]['level'] = df['font_size'].apply(lambda s: classify_font_size(s, font_to_level))
    return cleaned_dflist