"""
Author Andrey Vlasenko

Reads Wikipedia atricles specified in a variable "topics" and puts the content
into a list texts and saves it

"""

import wikipediaapi
import pickle
import re



def tokenize_expressions(text_list, unite_sentence=False):
    """
    Tokenize descriptions into expressions separated by commas.
    Args:
        text_list (list of str): List of text descriptions.
    Returns:
        list of list of str: Tokenized expressions for each description.
    """
    tokenized = [[expr.strip().lower() for expr in text.split(".")] for text in text_list]
    tokenized_flat = [ item +"." for sublist in tokenized for item in sublist]
    if unite_sentence: 
        new_tlist = [". "+ tokenized_flat[i] + " " + tokenized_flat[i + 1] for i in range(len(tokenized_flat) - 1)]
        return new_tlist
    else: 
        tokenized_flat = [ ". " + item for item in tokenized_flat]
        return tokenized_flat

def get_wikipedia_text(topic):
    """
    Fetches the text content of a Wikipedia page for the specified topic.
    
    Parameters:
        topic (str): The topic to search on Wikipedia.
    
    Returns:
        str: The flat text content of the Wikipedia page.
    """
    # Initialize Wikipedia API with a user agent
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="MyWikipediaFetcher/1.0 (https://mywebsite.com/; myemail@example.com)"
    )
    
    # Get the page for the specified topic
    page = wiki.page(topic)
    
    if page.exists():
        # Return the text content of the page as a single string
        return clean_text(page.text)
    else:
        raise ValueError(f"The Wikipedia page for '{topic}' does not exist.")
        
        
        
def clean_text(text):
    """
    Cleans text by removing line breaks (`\n`) and uniting words split by `-` at the end of lines.
    
    Parameters:
        text (str): The raw text to clean.
    
    Returns:
        str: The cleaned text.
    """
    # Remove line breaks
    text = text.replace("\n", " ")
    
    # Remove hyphen at line breaks and unite the words
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

        


topics = ["Charles Darwin", "Game of thrones", "Communism", "Red hat", "Capitalism", "fairy tail", "philosophy", 
          "Little Red Riding Hood", "history", "Feudalism", "Science", "Biology", "theater", "chemistry", 
          "geography", "literature", "Filmmaking", "Odyssey", "democracy", "slavery", "Europe", "Pinocchio", 
          "Snow White", "Electricity", "Rapunzel", "Robin Hood", "The Wonderful Wizard of Oz", "Magnetism", 
          "Crusades", "Evolution", "The Ugly Duckling", "The Art of War", "Religion", "Polytheism", 
          "Ancient Egypt", "The Roman Empire", "The French Revolution", "The Cold War", 
          "The Industrial Revolution", "The American Civil War", "The Renaissance", 
          "The Treaty of Versailles", "World War II", "The United Nations", "Quantum Mechanics", 
          "Artificial Intelligence", "Space Exploration", "Genetics", "Robotics", "The Theory of Relativity", 
          "Black Holes", "Environmental Science", "Renewable Energy", "Climate Change", "Greek Mythology", 
          "Norse Mythology", "The Legend of King Arthur", "The Trojan War", "Hercules", "The Mahabharata", 
          "The Epic of Gilgamesh", "Japanese Folklore", "Native American Mythology", "African Folklore", 
          "Existentialism", "Stoicism", "Utilitarianism", "Cognitive Behavioral Therapy", 
          "Freud and Psychoanalysis", "Jungian Archetypes", "Ethics", "Metaphysics", "Human Consciousness", 
          "Behavioral Economics", "Impressionism", "Surrealism", "The Canterbury Tales", 
          "Hamlet by William Shakespeare", "Don Quixote", "Frankenstein by Mary Shelley", "Romantic Poetry", 
          "Gothic Literature", "Modernism", "The Great Gatsby", "The Brothers Grimm", 
          "Hans Christian Andersen", "The Tale of Peter Rabbit", "The Chronicles of Narnia", 
          "Alice's Adventures in Wonderland", "The Jungle Book", "The Hobbit", "The Lord of the Rings", 
          "Aladdin and the Magic Lamp", "Arabian Nights", "The Silk Road", "Indigenous Peoples", 
          "Urbanization", "Pop Culture", "Music History", "Fashion through the Ages", "Food and Cuisine", 
          "The Internet Revolution", "Globalization", "Education Systems", "Astronomy", "Mathematics", 
          "Architecture", "Archaeology", "Cryptography", "History of Medicine", "Paleontology", 
          "Linguistics", "The History of Sports", "The Olympics"]



texts= []


for topic in topics:
    print("TOPIC = ", topic)
    try:
        text = get_wikipedia_text(topic)
        texts = texts +[text]
        print(f"Wikipedia content for '{topic}':\n{text[:500]}...")  
    except ValueError as e:
        print(e)


tlist = tokenize_expressions(texts, unite_sentence=True)

#tlist = [item for sublist in tokenized for item in sublist]
output_path = "/home/andrey/Downloads/Darwin/Darwin_biogr_list_large"

with open(output_path, "wb") as fp:   #Pickling
    pickle.dump(tlist, fp)



