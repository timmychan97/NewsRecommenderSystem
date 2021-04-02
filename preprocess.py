import pandas as pd
import dateutil.parser
import re


# Credit: https://github.com/stopwords-iso/stopwords-no/blob/master/raw/gh-stopwords-json-no.txt
NORWEGIAN_STOPWORDS = ["alle", "at", "av", "bare", "begge", "ble", "blei", "bli", "blir", "blitt", "både", "båe", "da", "de", "deg", "dei", "deim", "deira", "deires", "dem", "den", "denne", "der", "dere", "deres", "det", "dette", "di", "din", "disse", "ditt", "du", "dykk", "dykkar", "då", "eg", "ein", "eit", "eitt", "eller", "elles", "en", "enn", "er", "et", "ett", "etter", "for", "fordi", "fra", "før", "ha", "hadde", "han", "hans", "har", "hennar", "henne", "hennes", "her", "hjå", "ho", "hoe", "honom", "hoss", "hossen", "hun", "hva", "hvem", "hver", "hvilke", "hvilken", "hvis", "hvor", "hvordan", "hvorfor", "i", "ikke", "ikkje", "ingen", "ingi", "inkje", "inn", "inni", "ja", "jeg", "kan", "kom", "korleis", "korso", "kun", "kunne", "kva", "kvar", "kvarhelst", "kven", "kvi", "kvifor", "man", "mange", "me", "med", "medan", "meg", "meget", "mellom", "men", "mi", "min", "mine", "mitt", "mot", "mykje", "må", "måtte", "ned", "no", "noe", "noen", "noka", "noko", "nokon", "nokor", "nokre", "nå", "når", "og", "også", "om", "opp", "oss", "over", "på", "samme", "seg", "selv", "si", "sia", "sidan", "siden", "sin", "sine", "sitt", "sjøl", "skal", "skulle", "slik", "so", "som", "somme", "somt", "så", "sånn", "til", "um", "upp", "ut", "uten", "var", "vart", "varte", "ved", "vere", "verte", "vi", "vil", "ville", "vore", "vors", "vort", "vår", "være", "vært", "å"]

def find_good_keywords(keywords):
    """
    Remove all bad category candidates such as:
     - Norwegian stopwords
     - Numbers
     - Symbols
    
    Potential improvements:
     - Remove word endings such as "bil vs bilen", "treet vs tre", "løpe vs løper".
       This requires advanced NLP
       ...
    """
    # Lowercase all strings
    keywords = [kw.lower() for kw in keywords]

    # For each item, remove all characters except alphanumeric and accented characters
    keywords = map(lambda word: re.sub(r'[^A-Za-z0-9À-ÖØ-öø-ÿ]+', '', word), keywords)

    # Remove all items that is a number
    keywords = [kw for kw in keywords if not kw.isdigit()]

    # Remove all stopwords
    keywords = [kw for kw in keywords if not kw in NORWEGIAN_STOPWORDS]

    # Remove empty strings from list
    keywords = list(filter(None, keywords))
    return keywords
    

def derive_category_candidates(url):
    return (url[18:]        # Remove the http://adressa.no/ part
            .split("/")     # Get the routes
            [:-1])          # Don't take the last part  


def derive_document_details(df_docs):
    """
    Derive details of the documents (news articles) by different data we have
    """
    # Convert ISO 8601 datetime string to millis since 1970
    df_docs['publishtime'] = df_docs['publishtime'].apply(lambda x: dateutil.parser.parse(x).timestamp() if pd.notnull(x) else x)

    # Use the earliest time if publishtime is not present
    df_docs['publishtime'].fillna(df_docs['time'], inplace=True)

    # Derive and fill in categories from the url
    df_docs['category'] = df_docs.apply(lambda row: (
        "|".join(find_good_keywords(derive_category_candidates(row['url'])))
        if pd.isnull(row['category']) else row['category']
        ), axis=1)

    # Create new column "keywords" and fill it with words from category and title reasonably
    def derive_keywords(row):
        keywords = row['title'].split()
        keywords = find_good_keywords(keywords)
        if len(keywords) > 0:
            return row['category'] + '|' + '|'.join(keywords)
        else:
            return row['category']

    df_docs['keywords'] = df_docs.apply(derive_keywords, axis=1)
    
    # Drop url, time and title columns, we don't need them anymore
    df_docs.drop(['url', 'time', 'title'], axis=1, inplace=True)
    
    
def analyze_documents(df):
    """
    Loop the dataset and find all available information about the documents,
    and make another dataset for easier access.

    Adressa dataset details:
     - `publishtime` is written in ISO 8601 format, and `time` is written in millis
     - `publishtime` could be missing, We can derive it from the earliest found interaction
        - 5451 NaN publishtime values
     - category could be missing, but the URL does not miss. We can derive the categories from the URL
        - 7615 NaN category values
    """
    total_num = df.shape[0]
    df_docs = df[df['documentId'].notnull()].copy()

    df_docs.sort_values(by=['documentId', 'time'], ascending=True, inplace=True)
    df_docs.drop_duplicates(subset='documentId', keep='first', inplace=True)
    df_docs = df_docs[['time','documentId','publishtime','category','title','url']]
    derive_document_details(df_docs)

    df_docs.to_csv('documents.csv', sep=',', index=False, encoding='utf-8-sig')