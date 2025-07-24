import re

def clean_string(tinput):
    # Remove all byte/hex-like markers: <..>, ^.., etc.
    cleaned = re.sub(r'<[^>]{1,4}>', ' ', tinput)  # Remove <...> blocks
    cleaned = re.sub(r'\^[A-Za-z0-9]', ' ', cleaned)  # Remove ^D, ^A, etc.
    cleaned = re.sub(r'[^\x20-\x7E\n\.]', ' ', cleaned)  # Remove non-ascii except newlines and periods
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Collapse whitespace
    return cleaned.strip()



def clean_text(raw):

    return [clean_string(item) for item in raw]

