
def recalculate_word_stats(dflist):
    """
    Recalculates word count and word density (words per font size)
    for each row in each DataFrame of the given list.

    Parameters:
        dflist (list of pd.DataFrame): Each DataFrame should have at least
                                       'text' and 'font_size' columns.

    Returns:
        list of pd.DataFrame: Updated DataFrames with recalculated columns:
                              'word_count' and 'word_density'.
    """
    updated_dflist = []

    for df in dflist:
        if 'text' not in df.columns or 'font_size' not in df.columns:
            updated_dflist.append(df)  # skip pages without required info
            continue

        df = df.copy()
        df['word_count'] = df['text'].apply(lambda t: len(str(t).split()))
        df['word_density'] = df.apply(
            lambda row: row['word_count'] / row['font_size'] if row['font_size'] else 0,
            axis=1
        )
        updated_dflist.append(df)

    return updated_dflist


def analyze_word_density_patterns(dflist, method='word_density', stat='average', threshold_factor=1.5):
    """
    Analyzes word density in a list of DataFrames to find lines that are unusually dense or sparse.

    Parameters:
        dflist (list of pd.DataFrame): List of page DataFrames with 'word_count', 'font_size', 'line_height'.
        method (str): Mode of analysis. One of:
            - 'word_density'     : word_count / line_height
            - 'density_by_font'  : word_count / font_size
            - 'weighted_density' : (word_count ** 2) / font_size
            - 'font_scaled'      : word_density * font_size
        stat (str): How to compute the central reference ('average', 'median').
        threshold_factor (float): How far from the mean is considered rare/unusual.

    Returns:
        dict: {
            'metric_name': str,
            'central_value': float,
            'dense_lines': list of (page_idx, row_idx),
            'sparse_lines': list of (page_idx, row_idx)
        }
    """
    scores = []
    locations = []

    for page_idx, df in enumerate(dflist):
        for row_idx, row in df.iterrows():
            wc, fs, lh = row.get('word_count'), row.get('font_size'), row.get('line_height')
            if not all([wc, fs, lh]) or fs == 0 or lh == 0:
                continue

            # Compute metric based on mode
            if method == 'word_density':
                score = wc / lh
            elif method == 'density_by_font':
                score = wc / fs
            elif method == 'weighted_density':
                score = (wc ** 2) / fs
            elif method == 'font_scaled':
                score = (wc / lh) * fs
            else:
                raise ValueError(f"Invalid method: {method}")

            scores.append(score)
            locations.append((page_idx, row_idx))

    scores = np.array(scores)

    # Central tendency
    if stat == 'average':
        central = np.mean(scores)
    elif stat == 'median':
        central = np.median(scores)
    else:
        raise ValueError("Stat must be 'average' or 'median'")

    lower_thresh = central / threshold_factor
    upper_thresh = central * threshold_factor

    # Collect dense/sparse locations
    dense_lines = [loc for score, loc in zip(scores, locations) if score > upper_thresh]
    sparse_lines = [loc for score, loc in zip(scores, locations) if score < lower_thresh]

    return {
        'metric_name': method,
        'central_value': round(central, 3),
        'dense_lines': dense_lines,
        'sparse_lines': sparse_lines
    }