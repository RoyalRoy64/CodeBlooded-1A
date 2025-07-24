import pymupdf
import pymupdf4llm
import pathlib
import pandas as pd
from pprint import pprint
# pip install llama_index
fileinquestion = "C1A/input/E0CCG5S239.pdf"
doc = pymupdf.open(fileinquestion)  # or pymupdf.Document(filename)
print(doc.page_count)
print(doc.metadata)
print(doc.get_toc())
# page = doc.load_page(2)
# page = doc[1] #not index, its page
# for page in reversed(doc):
# for page in doc.pages(start, stop, step):
for i,page in enumerate(doc): # iterate the document pages
    with open(f"analysis/page{i}.txt", "w") as f:
        # f.write(page.text)
        pass
    text = page.get_text() # get plain text encoded as UTF-8
    # print(text)
    with open(f"analysis/gettext{i}.txt", "w") as f:
        f.write(text)    
    links = page.get_links() 
    link = page.first_link  # a `Link` object or `None`
    while link: 
        print(type(link))
        link = link.next # get next link, last one has `None` in its `next`
    for annot in page.annots():
        print(f'Annotation on page: {page.number} with type: {annot.type} and rect: {annot.rect}')
    for field in page.widgets():
        print(f'Widget on page: {page.number} with type: {field.type} and rect: {field.rect}')

    tabs = page.find_tables() # locate and extract any tables on page
    print(f"{len(tabs.tables)} found on {page}") # display number of found tables
    print(type(tabs.tables))
    print(tabs[0].extract())
    check =pd.DataFrame(tabs[0].extract())
    if tabs.tables:  # at le    ast one table found?
        # pprint(tabs[0].extract())  # print content of first table
        pass
    # text = page.get_text("html")
        # Use one of the following strings for opt to obtain different formats [2]:
        # “text”: (default) plain text with line breaks. No formatting, no text position details, no images.
        # “blocks”: generate a list of text blocks (= paragraphs).
        # “words”: generate a list of words (strings not containing spaces).
        # “html”: creates a full visual version of the page including any images. This can be displayed with your internet browser.
        # “dict” / “json”: same information level as HTML, but provided as a Python dictionary or resp. JSON string. See TextPage.extractDICT() for details of its structure.
        # “rawdict” / “rawjson”: a super-set of “dict” / “json”. It additionally provides character detail information like XML. See TextPage.extractRAWDICT() for details of its structure.
        # “xhtml”: text information level as the TEXT version but includes images. Can also be displayed by internet browsers.
        # “xml”: contains no images, but full position and font information down to each single text character. Use an XML module to interpret.
    # with open("output.html", "w") as f:
    #     f.write(text)

md_text = pymupdf4llm.to_markdown(fileinquestion)
pathlib.Path("output.md").write_bytes(md_text.encode())
# llama_reader = pymupdf4llm.LlamaMarkdownReader()
# print(llama_reader())
# llama_docs = llama_reader.load_data(fileinquestion)
