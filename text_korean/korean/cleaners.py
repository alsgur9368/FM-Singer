import re

def korean_cleaners(text):

    text = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z,.!?~…_\s\-]", "", text)
    
    return text
