import pandas as pd
def find_alternate_page_repeats_split(dflist, mid, tolerance=1.0, match_all=True):
    """
    Finds repeated elements on alternating pages (e.g. headers/footers).
    
    Parameters:
        dflist (list of pd.DataFrame): List of pages as DataFrames.
        mid (int): A central page index to start comparison from.
        tolerance (float): Vertical `top` position tolerance.
        match_all (bool): If True, text must appear in all matching pages (stricter). If False, any one match is enough (looser).

    Returns:
        even_matches, odd_matches (DataFrames)
    """
    def is_valid_df(df):
        return df is not None and not df.empty

    def get_matches(df_base, *others):
        matched = []
        for _, row in df_base.iterrows():
            text, top = row['text'].strip(), row['top']

            def match(df):
                return any((abs(top - r['top']) <= tolerance and r['text'].strip() == text) for _, r in df.iterrows())

            if (match_all and all(match(df) for df in others if is_valid_df(df))) or \
               (not match_all and any(match(df) for df in others if is_valid_df(df))):
                matched.append(row)

        return pd.DataFrame(matched)

    def get_neighbors(idx):
        before = dflist[idx - 2] if idx - 2 >= 0 else None
        after  = dflist[idx + 2] if idx + 2 < len(dflist) else None
        return [df for df in [before, after] if is_valid_df(df)]

    even_matches = get_matches(dflist[mid], *get_neighbors(mid)) if 0 <= mid < len(dflist) else pd.DataFrame()
    odd_idx = mid - 1
    odd_matches = get_matches(dflist[odd_idx], *get_neighbors(odd_idx)) if 0 <= odd_idx < len(dflist) else pd.DataFrame()

    return even_matches.reset_index(drop=True), odd_matches.reset_index(drop=True)

def clean_pages_of_even_odd_repeats(dflist, even_df, odd_df, tolerance=1.0):
    """
    Cleans even-indexed pages using even_df and odd-indexed pages using odd_df.
    If even_df == odd_df, treats all pages the same.
    
    Returns:
        - cleaned_pages: list of cleaned DataFrames
        - header_footer_removed_pages: list of page indices that had header/footer removed
    """
    def to_clean_set(df):
        return {(row['text'].strip(), round(row['top'], 1)) for _, row in df.iterrows() if isinstance(row['text'], str)}

    even_set = to_clean_set(even_df)
    odd_set = to_clean_set(odd_df)
    same_repeats = even_set == odd_set

    cleaned_pages = []
    header_footer_removed_pages = []

    for idx, df in enumerate(dflist):
        if df.empty or df is None:
            cleaned_pages.append(df)
            continue

        target_set = even_set if same_repeats or idx % 2 == 0 else odd_set

        # Mark redundant lines (possible headers/footers)
        df['is_redundant'] = df.apply(
            lambda row: any(
                isinstance(row['text'], str) and
                abs(row['top'] - rep_top) <= tolerance and
                row['text'].strip() == rep_text
                for rep_text, rep_top in target_set
            ), axis=1
        )

        if df['is_redundant'].any():
            header_footer_removed_pages.append(idx)

        cleaned_df = df[~df['is_redundant']].drop(columns='is_redundant').reset_index(drop=True)
        cleaned_pages.append(cleaned_df)

    return cleaned_pages, header_footer_removed_pages
