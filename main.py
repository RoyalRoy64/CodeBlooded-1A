import pymupdf
# import pymupdf4llm
import pathlib
import pandas as pd
import numpy as np
from pprint import pprint
import re
from collections import Counter
from extractandclean    import extract_blocks_and_distributions , extract_heading_summary , extract_crucial_pattern_lines , filter_and_clean_gibberish
from merge              import merge_consecutive_same_font_and_left
from headerfooter       import find_alternate_page_repeats_splitv2 , find_alternate_page_repeats_split , clean_pages_of_even_odd_repeats
from tables             import extract_table_with_title
from fontstat           import generate_font_stats , get_global_font_counter , get_rare_large_fonts , map_fonts_to_heading_levels , get_significant_large_fonts , create_font_level_map , get_top_fonts , classify_font_size
from analyzeandcompute  import unify_close_left_values , analyze_left_distribution , analyze_left_clusters , analyze_line_spacing , compute_word_count_stats , compute_line_spacing_per_page
# md_text = pymupdf4llm.to_markdown(fileinquestion)
# pathlib.Path("output.md").write_bytes(md_text.encode())
# llama_reader = pymupdf4llm.LlamaMarkdownReader()
# print(llama_reader())
# llama_docs = llama_reader.load_data(fileinquestion)

#tesseract ocr
# english
# russian
# chinese (new)
# mandarin (traditional)
# japanese
# Hindi
# Arabic
# French
# Hebrew
# German
# Korean
# Italian
# Polish
# Portugese
# Spanish 
# Indonesian3
#turkish
# Urdu
#pip install pymupdf numpy pandas sentence-transformers
#################### workflow
#parallel processing
#trigger - 
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

fileinquestion = "C1A/input/E0CCG5S239.pdf" #single page doc E0CCG5S239
fileinquestion = "C1A/input/TOPJUMP-PARTY-INVITATION-20161003-V01.pdf"
fileinquestion = "C1A/input/STEMPathwaysFlyer.pdf"
fileinquestion = "CustomPDFs/killer.pdf"
fileinquestion = "CustomPDFs/jess401.pdf"
fileinquestion = "C1A/input/E0CCG5S312.pdf"   #easy
fileinquestion = "C1A/input/E0H1CM114.pdf" ######### hard
doc = pymupdf.open(fileinquestion)  
page = doc[0]  # first page
width , height = page.rect.width , page.rect.height
doctoc = doc.get_toc()
docmetadata = doc.metadata
totalpage= doc.page_count
dflist   = []
mid = totalpage//2

for i in range(totalpage):
    page = doc.load_page(i)
    html = page.get_text("html")
    blocks= extract_blocks_and_distributions(html)
    dflist.append(pd.DataFrame(blocks))############### to dict if we do async
dfl = dflist
even_df, odd_df =   find_alternate_page_repeats_splitv2(dflist, mid=totalpage // 2)
dflist          =   clean_pages_of_even_odd_repeats(dflist, even_df, odd_df)
dflist          =   [filter_and_clean_gibberish(df, min_alnum_ratio=0.15, min_length=1) for df in dflist] 
averagewords    =   compute_word_count_stats(dflist)['global_avg_words_per_line']
font_stats_df   =   generate_font_stats(dflist)

dflist = unify_close_left_values(dflist, tolerance=width*0.01, min_left=0, max_left=width)

################add it back
#['page_number', 'font_size', 'font_family', 'count', 'total_words','avg_word_density']
# fontfamily_counter = Counter(font_stats_df['font_family'])
# fontsize_counter = Counter(font_stats_df['font_size'])

font_word_counts = (font_stats_df.groupby('font_size')['total_words'].sum().sort_values(ascending=False) )
# master_df = combine_dflist_to_master_df(dflist)
target_fonts = get_significant_large_fonts(font_word_counts, z_thresh=0.5)['large_fonts']
# print("Large fonts:", result['large_fonts']) #print("Z-scores:\n", result['z_scores']) #print("Body font:", result['body_font'])

spacing_list = compute_line_spacing_per_page(dflist)
all_spacings = np.concatenate(spacing_list)# If you want to concatenate all spacing into a single array:
spacing_stats = analyze_line_spacing(spacing_list)
meansp =  spacing_stats['mean_spacing']

dflist = merge_consecutive_same_font_and_left(dflist,list(font_word_counts.keys()),[meansp*0.4,meansp*1.75])
leftdf = analyze_left_distribution(dflist)['left_stats_df'] # 'left_counter': left_counter,# 'dominant_lefts': dominant_lefts,# 'left_stats_df': left_stats_df
leftdf.loc[(leftdf.left<=204) & (leftdf.percentage>=1)]

master_df = combine_dflist_to_master_df(dflist)

column_order = ['page_number',  'left',"top", 'end' ,"text",'line_height','font_size','font_family','bold','italic','word_count','word_density','parabool']
master_df['left'] = master_df['left'].round(2)

sortmm = (master_df.loc[master_df.left<width*0.35])
unique_lefts = sorted(set(val for val in sortmm['left'].unique()))



df_summary = analyze_left_clusters([master_df], unique_lefts) #uniql are ascending
testingdf = (master_df.loc[(master_df.left==unique_lefts[0])& (master_df.word_count<=10)][['top','line_height','font_size','bold','italic','text','word_count','parabool']])
print(font_word_counts)



HEADING_PATTERNS = [
    r"^\d+\.",                # 1.
    r"^\d+\.\d+(\.\d+)*",  # 1.1, 2.3.4
    r"^\d+\)",               # 1)
    r"^[A-Z]\.",             # A.
    r"^[a-z]\)",             # a)
    r"^[ivxlcdm]+\)",        # i), ii), iv)
    r"^[IVXLCDM]+\)",        # I), II)
    r"^[a-z]\.",             # a.
    r"^[IVXLCDM]+\.",        # I.
    r"^\u2022\s*",          # • bullet
    r"^-\s*",                # - bullet
    r"^\*\s*",              # * bullet
]

HEADING_REGEX = re.compile("|".join(HEADING_PATTERNS))


def extract_heading_candidates(df, body_font_size=11.0, max_words=10):
    """
    Extract likely heading lines from a DataFrame based on font size, boldness, left alignment,
    and text content pattern.

    Parameters:
        df (pd.DataFrame): The input DataFrame with line-level text data.
        body_font_size (float): Estimated body font size threshold.
        max_words (int): Maximum word count for heading candidates.

    Returns:
        pd.DataFrame: Filtered DataFrame containing heading candidates with a new column 'heading_level'.
    """
    df = df.copy()
    
    # Apply pattern-based heading detection
    def is_heading_pattern(text):
        return bool(HEADING_REGEX.match(text.strip()))

    df['is_heading_pattern'] = df['text'].astype(str).apply(is_heading_pattern)

    # Basic rule-based heading detection
    df['heading_candidate'] = (
        (df['font_size'] > body_font_size) |
        (df['bold']) |
        (df['italic']) |
        (df['is_heading_pattern'])
    ) & (df['word_count'] <= max_words)

    heading_df = df[df['heading_candidate']].copy()

    # Assign heading levels dynamically
    grouped = heading_df.groupby(['font_size', 'bold', 'italic'])
    groups = sorted(grouped.groups.keys(), key=lambda k: (-k[0], -int(k[1]), -int(k[2])))

    # Heading level assignment logic
    level_map = {}
    levels = ['H1', 'H2', 'H3']
    for i, group_key in enumerate(groups[:6]):
        if len(groups) <= 3:
            level_map[group_key] = levels[i]
        elif len(groups) == 4:
            level_map[group_key] = levels[0] if i < 2 else levels[2]
        elif len(groups) == 5:
            level_map[group_key] = levels[0] if i == 0 else (levels[1] if i < 3 else levels[2])
        elif len(groups) >= 6:
            level_map[group_key] = levels[i // 2]  # two each

    heading_df['heading_level'] = heading_df.apply(
        lambda row: level_map.get((row['font_size'], row['bold'], row['italic']), 'H3'), axis=1
    )

    return heading_df.reset_index(drop=True)


heading_candidatesL1 = extract_heading_candidates(testingdf, body_font_size=11.0, max_words=10)
heading_candidatesL2 = extract_heading_candidates(testingdf, body_font_size=11.0, max_words=10)
heading_candidatesL3 = extract_heading_candidates(testingdf, body_font_size=11.0, max_words=10)

print(heading_candidates)