import pandas as pd
def find_alternate_page_repeats_splitv2(dflist, mid):
    def is_valid_df(df):
        return df is not None and not df.empty

    def get_matches(df_base, *others):
        matched = []
        for _, row in df_base.iterrows():
            text = row['text'].strip()
            top = row['top']

            def match(df):
                return any(
                    (abs(top - r['top']) <= 2.0) and (r['text'].strip() == text)
                    for _, r in df.iterrows()
                )

            if any(is_valid_df(df) and match(df) for df in others):
                matched.append(row)

        return pd.DataFrame(matched)

    even_matches, odd_matches = pd.DataFrame(), pd.DataFrame()

    # Process current page (mid) regardless of parity
    if 0 <= mid < len(dflist):
        page = dflist[mid]
        before = dflist[mid - 2] if mid - 2 >= 0 else None
        after = dflist[mid + 2] if mid + 2 < len(dflist) else None

        if is_valid_df(before) or is_valid_df(after):
            even_matches = get_matches(page, *(df for df in [before, after] if is_valid_df(df)))

    # Process previous page (mid - 1)
    odd_page_idx = mid - 1
    if 0 <= odd_page_idx < len(dflist):
        page = dflist[odd_page_idx]
        before = dflist[odd_page_idx - 2] if odd_page_idx - 2 >= 0 else None
        after = dflist[odd_page_idx + 2] if odd_page_idx + 2 < len(dflist) else None

        if is_valid_df(before) or is_valid_df(after):
            odd_matches = get_matches(page, *(df for df in [before, after] if is_valid_df(df)))

    return even_matches.reset_index(drop=True), odd_matches.reset_index(drop=True)

def find_alternate_page_repeats_split(dflist, mid):
    def is_valid_df(df):
        return df is not None and not df.empty

    def get_matches(df_base, *others):
        matched = []
        for _, row in df_base.iterrows():
            text = row['text'].strip()
            top = row['top']

            def match(df):
                return any(
                    (abs(top - r['top']) <= 1.0) and (r['text'].strip() == text)
                    for _, r in df.iterrows()
                )

            if all(match(df) for df in others if is_valid_df(df)):
                matched.append(row)

        return pd.DataFrame(matched)

    even_matches, odd_matches = pd.DataFrame(), pd.DataFrame()

    # Even page block
    if mid % 2 == 0 and mid < len(dflist):
        page_even = dflist[mid]
        before_even = dflist[mid - 2] if mid - 2 >= 0 else None
        after_even = dflist[mid + 2] if mid + 2 < len(dflist) else None

        if is_valid_df(before_even) and is_valid_df(after_even):
            even_matches = get_matches(page_even, before_even, after_even)
        elif is_valid_df(before_even):
            even_matches = get_matches(page_even, before_even)
        elif is_valid_df(after_even):
            even_matches = get_matches(page_even, after_even)

    # Odd page block
    odd_page_idx = mid - 1
    if odd_page_idx >= 0 and odd_page_idx % 2 == 1 and odd_page_idx < len(dflist):
        page_odd = dflist[odd_page_idx]
        before_odd = dflist[odd_page_idx - 2] if odd_page_idx - 2 >= 0 else None
        after_odd = dflist[odd_page_idx + 2] if odd_page_idx + 2 < len(dflist) else None

        if is_valid_df(before_odd) and is_valid_df(after_odd):
            odd_matches = get_matches(page_odd, before_odd, after_odd)
        elif is_valid_df(before_odd):
            odd_matches = get_matches(page_odd, before_odd)
        elif is_valid_df(after_odd):
            odd_matches = get_matches(page_odd, after_odd)

    return even_matches.reset_index(drop=True), odd_matches.reset_index(drop=True)



def clean_pages_of_even_odd_repeats(dflist, even_df, odd_df, tolerance=1.0):
    """
    Cleans even-indexed pages using even_df and odd-indexed pages using odd_df.
    If even_df == odd_df, treats all pages the same.
    """
    def to_clean_set(df):
        return set((row['text'].strip(), round(row['top'], 1)) for _, row in df.iterrows())

    even_set = to_clean_set(even_df)
    odd_set = to_clean_set(odd_df)
    same_repeats = even_set == odd_set

    cleaned_pages = []

    for idx, df in enumerate(dflist):
        compare_set = even_set if same_repeats or idx % 2 == 0 else odd_set

        mask = df.apply(
            lambda row: not any(
                (abs(row['top'] - rep_top) <= tolerance) and (row['text'].strip() == rep_text)
                for rep_text, rep_top in compare_set
            ),
            axis=1
        )
        cleaned_df = df[mask].reset_index(drop=True)
        cleaned_pages.append(cleaned_df)
    return cleaned_pages