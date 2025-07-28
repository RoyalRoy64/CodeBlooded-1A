import re
import pandas as pd
import html
heading_patterns = [
    r'^\d+\.',                # 1.
    r'^\d+\.\d+(\.\d+)*',     # 1.1, 2.3.4
    r'^\d+\)',                # 1)
    r'^[A-Z]\.',              # A.
    r'^[a-z]\)',              # a)
    r'^[ivxlcdm]+\)',         # i), ii), iv)
    r'^[IVXLCDM]+\)',         # I), II)
    r'^[a-z]\.',              # a.
    r'^[IVXLCDM]+\.',         # I.
    r'^•\s*',                 # • bullet
    r'^-\s*',                 # - bullet
    r'^\*\s*'                 # * bullet
]
compiled_patterns = [re.compile(p) for p in heading_patterns]



def extract_blocks_and_distributions(html_string): # Match each full <p> block
    block_pattern = re.findall(
        r'<p style="top:([\d.]+)pt;left:([\d.]+)pt;line-height:([\d.]+)pt">(.*?)</p>',
        html_string,
        flags=re.DOTALL
    )
    results , font_sizes, line_heights = [],[],[]

    for top, left, line_height, inner_html in block_pattern:
        font_match = re.search(r'font-size:([\d.]+)pt', inner_html)
        font_size = float(font_match.group(1)) if font_match else None

        font_family_match = re.search(r'font-family:([^;"]+)', inner_html)
        font_family = font_family_match.group(1).strip() if font_family_match else None

        is_bold = bool(re.search(r'<b>', inner_html))
        is_italic = bool(re.search(r'<i>', inner_html))

        clean_text = re.sub(r'<[^>]+>', '', inner_html).strip()
        word_count = len(clean_text.split())
        word_density = word_count / float(line_height) if float(line_height) > 0 else 0

        results.append({
            "top": float(top),
            "left": float(left),
            "line_height": float(line_height),
            "font_size": font_size,
            "font_family": font_family,
            "bold": is_bold,
            "italic": is_italic,
            "text": clean_text,
            "word_count": word_count,
            "word_density": word_density
        })
        line_heights.append(float(line_height))
        font_size = float(font_match.group(1))
        font_sizes.append(font_size)

        line_heights.append(float(line_height))
    if results:
        return pd.DataFrame(results)# ,columns=["top", "left", "line_height", "font_size", "font_family","bold", "italic", "text", "word_count", "word_density"])
    else:
        return pd.DataFrame(columns=["top", "left", "line_height", "font_size", "font_family","bold", "italic", "text", "word_count", "word_density"])

def extract_crucial_pattern_lines(dflist):
    dfcrucial_pattern = []
    for page_num, df in enumerate(dflist):
        if df.empty or 'text' not in df.columns:
            continue

        for _, row in df.iterrows():
            first_word = row['text'].strip().split(' ')[0]
            for pattern in compiled_patterns:
                if pattern.match(first_word):
                    row_copy = row.copy()
                    row_copy['page'] = page_num
                    dfcrucial_pattern.append(row_copy)
                    break  # stop at first match
    if dfcrucial_pattern:
        return pd.DataFrame(dfcrucial_pattern).reset_index(drop=True)
    else:
        return pd.DataFrame()
    
def extract_heading_summary(cleaned_dflist, levels=('H1', 'H2')):
    """Extracts rows marked as H1 or H2 and returns a summary DataFrame."""
    heading_rows = []
    for page_num, df in enumerate(cleaned_dflist):
        if 'level' not in df.columns:
            continue
        headings_df = df[df['level'].isin(levels)].copy()
        headings_df['page_num'] = page_num
        heading_rows.append(headings_df[['page_num', 'level', 'text']])

    return pd.concat(heading_rows, ignore_index=True) if heading_rows else pd.DataFrame()

def filter_and_clean_gibberish(
    df,
    text_column='text',
    min_alnum_ratio=0.1,
    min_length=1,
    preserve_tokens=None
):
    """
    Cleans gibberish from a DataFrame column by:
    - Preserving structured patterns like '3.1', 'v2.0', '1.', 'A.', etc.
    - Removing tokens with low alphanumeric content
    - Replacing gibberish with space, and dropping rows with no meaningful text

    Parameters:
        df (pd.DataFrame): The DataFrame with a column to clean.
        text_column (str): The column name containing text.
        min_alnum_ratio (float): Minimum ratio of alphanumeric characters per token.
        min_length (int): Minimum length of cleaned text to retain the row.
        preserve_tokens (set): Extra tokens to preserve, e.g., {'--', 'N/A', '%'}.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if preserve_tokens is None:
        preserve_tokens = {"-", "--", "---", "N/A", "n.a", "N.A.", "%", "—", "/", ":"}

    # Matches known valid structured tokens: 1., 3.1, v2.0, etc.
    structured_pattern = re.compile(r'^(v)?\d+(\.\d+)*\.?$|^[a-zA-Z]\.$|^[a-zA-Z]\)$')

    def is_meaningful(token):   
        token = token.strip()
        if not token:
            return False

        if token in preserve_tokens:
            return True

        if structured_pattern.match(token):
            return True

        alnum_ratio = sum(c.isalnum() for c in token) / (len(token) + 1e-6)
        return alnum_ratio >= min_alnum_ratio

    def clean_text(text):
        text = str(text).strip()
        if not text or len(text) < min_length:
            return None

        # Split on whitespace only (preserve punctuation within tokens like 3.1)
        raw_tokens = re.findall(r'\S+', text)
        cleaned_tokens = []

        for token in raw_tokens:
            if is_meaningful(token):
                cleaned_tokens.append(token)
            else:
                cleaned_tokens.append(' ')  # replace junk with space

        cleaned = ' '.join(cleaned_tokens).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)

        if len(cleaned) < min_length or all(c in "-*.,;: " for c in cleaned):
            return None

        return cleaned
    if df.empty or []:
        return df
    df = df.copy()
    df[text_column] = df[text_column].apply(clean_text)
    df = df[df[text_column].notna()].reset_index(drop=True)
    return df

def decode_text_in_dflist_to_utf8(dflist, text_column='text'):
    """
    Decodes HTML entities (e.g., &#x414;) in the specified text column of each DataFrame in dflist.
    
    Parameters:
        dflist (list of pd.DataFrame): List of DataFrames representing pages.
        text_column (str): The name of the column to decode. Defaults to 'text'.
    
    Returns:
        list of pd.DataFrame: Updated list with decoded text content.
    """
    updated_dflist = []
    
    for df in dflist:
        if df.empty:
            updated_dflist.append(df)
            continue
        df = df.copy()
        if text_column in df.columns:
            df[text_column] = df[text_column].astype(str).apply(html.unescape)
        updated_dflist.append(df)
    
    return updated_dflist