import pandas as pd
def merge_consecutive_same_font_and_left(dflist, target_fonts, acceptable_spacing_range=[0.0, 2.0]):
    """
    Merges lines within each DataFrame in dflist where:
    - font_size ∈ target_fonts
    - consecutive lines share same font_size, left alignment, and bold/italic state
    - vertical spacing (top difference) is within acceptable_spacing_range

    Adds/Updates:
    - 'parabool' column: True if line is a merged paragraph.
    - 'end' column: set to its own top if no merge, or the last merged line's top.
    - 'word_count': updated count after merge
    - 'word_density': word_count / line_height

    Parameters:
        dflist (list of pd.DataFrame): Each DataFrame is a page.
        target_fonts (list): Font sizes to consider for merging.
        acceptable_spacing_range (list): [min_spacing, max_spacing] range to allow merging lines.

    Returns:
        list of pd.DataFrame: Modified DataFrames with merged paragraphs and updated columns.
    """
    min_spacing, max_spacing = acceptable_spacing_range
    updated_dflist = []

    for df in dflist:
        if df.empty or 'left' not in df.columns or 'font_size' not in df.columns:
            df['parabool'] = False
            df['end'] = df['top'] if 'top' in df.columns else 0
            df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
            df['word_density'] = df.apply(lambda row: row['word_count'] / row['line_height'] if row['line_height'] else 0, axis=1)
            updated_dflist.append(df)
            continue

        df = df.copy()
        df['parabool'] = False
        df['end'] = df['top']
        drop_indices = set()

        for left_value in sorted(df['left'].unique()):
            group_df = df[(df['left'] == left_value) & (df['font_size'].isin(target_fonts))]
            indices = group_df.index.tolist()
            i = 0

            while i < len(indices):
                current_idx = indices[i]
                current_font = df.loc[current_idx, 'font_size']
                current_bold = df.loc[current_idx, 'bold']
                current_italic = df.loc[current_idx, 'italic']
                current_top = df.loc[current_idx, 'top']
                final_end = current_top

                j = i + 1
                merged = False

                while j < len(indices):
                    next_idx = indices[j]
                    next_font = df.loc[next_idx, 'font_size']
                    next_bold = df.loc[next_idx, 'bold']
                    next_italic = df.loc[next_idx, 'italic']
                    next_top = df.loc[next_idx, 'top']

                    if (
                        next_font != current_font or
                        next_bold != current_bold or
                        next_italic != current_italic
                    ):
                        break

                    top_diff = abs(next_top - current_top)
                    if not (min_spacing <= top_diff <= max_spacing):
                        break

                    df.at[current_idx, 'text'] += ' ' + df.loc[next_idx, 'text']
                    final_end = next_top
                    drop_indices.add(next_idx)
                    merged = True
                    j += 1
                    current_top = next_top

                df.at[current_idx, 'parabool'] = merged
                df.at[current_idx, 'end'] = final_end
                i = j

        df = df.drop(index=list(drop_indices)).reset_index(drop=True)

        # Recalculate word_count and word_density
        df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
        df['word_density'] = df.apply(lambda row: row['word_count'] / row['line_height'] if row['line_height'] else 0, axis=1)

        updated_dflist.append(df)

    return updated_dflist


def combine_dflist_to_master_df(dflist):
    """
    Combines a list of page-level DataFrames into a single master DataFrame.
    Adds a 'page_number' column to indicate the source page for each row.
    Parameters:
        dflist (list of pd.DataFrame): A list where each DataFrame corresponds to one PDF page.
    Returns:
        pd.DataFrame: Combined DataFrame with an added 'page_number' column.
    """
    combined_rows = []
    for page_number, df in enumerate(dflist):
        if df is not None and not df.empty:
            df_copy = df.copy()
            df_copy['page_number'] = page_number
            combined_rows.append(df_copy)

    if not combined_rows:
        return pd.DataFrame()  # Return empty DataFrame if nothing valid

    master_df = pd.concat(combined_rows, ignore_index=True)
    return master_df
