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



new_topics = [
    "Isaac Newton", "Albert Einstein", "Galileo Galilei", "Marie Curie", "Nikola Tesla",
    "Stephen Hawking", "Gregor Mendel", "Ada Lovelace", "Alan Turing", "Rosalind Franklin",
    "DNA", "Mitochondria", "Photosynthesis", "Plate Tectonics", "Continental Drift",
    "Volcanoes", "Earthquakes", "Tornadoes", "Hurricanes", "Rainforests",
    "Deserts", "Mount Everest", "The Amazon River", "The Sahara Desert", "The Great Barrier Reef",
    "Antarctica", "The North Pole", "The South Pole", "The Pacific Ocean", "The Atlantic Ocean",
    "The Indian Ocean", "The Arctic Ocean", "The Mediterranean Sea", "The Black Sea", "The Caspian Sea",
    "The Dead Sea", "The Himalayas", "The Andes", "The Alps", "The Rockies",
    "The Appalachian Mountains", "The Mississippi River", "The Nile River", "The Yangtze River", "The Ganges River",
    "The Thames", "The Seine", "The Danube", "The Rhine", "The Amazon Rainforest",
    "Coral Reefs", "Mangroves", "Savannas", "Tundra", "Taiga",
    "Biodiversity", "Endangered Species", "Extinction", "Conservation", "Deforestation",
    "Pollution", "Greenhouse Effect", "Ozone Layer", "Acid Rain", "Sustainable Development",
    "Recycling", "Composting", "Solar Power", "Wind Power", "Hydroelectric Power",
    "Geothermal Energy", "Fossil Fuels", "Oil Spills", "Nuclear Power", "Radioactivity",
    "The Periodic Table", "Atoms", "Molecules", "Chemical Reactions", "Organic Chemistry",
    "Inorganic Chemistry", "Physical Chemistry", "Analytical Chemistry", "Biochemistry", "Thermodynamics",
    "Kinetics", "Quantum Physics", "Classical Mechanics", "Electromagnetism", "Optics",
    "Acoustics", "Thermodynamics (Physics)", "Statistical Mechanics", "Nuclear Physics", "Particle Physics",
    "String Theory", "The Big Bang Theory", "Cosmology", "The Milky Way", "The Solar System",
    "The Sun", "The Moon", "Mars", "Jupiter", "Saturn",
    "Uranus", "Neptune", "Pluto", "Comets", "Asteroids",
    "Meteorites", "Space Probes", "Satellites", "International Space Station", "Space Telescopes",
    "Hubble Space Telescope", "Voyager Program", "Apollo Program", "Mars Rover", "Space Shuttle",
    "Rocketry", "Orbital Mechanics", "Gravity", "Blackbody Radiation", "Dark Matter",
    "Dark Energy", "Exoplanets", "Life on Other Planets", "SETI", "Astrobiology",
    "The Human Genome Project", "Stem Cells", "CRISPR", "Gene Therapy", "Vaccines",
    "Pandemics", "The Plague", "The Spanish Flu", "COVID-19", "Antibiotics",
    "Bacteria", "Viruses", "Fungi", "Protozoa", "Algae",
    "The Nervous System", "The Brain", "Neurons", "Synapses", "Neurotransmitters",
    "The Endocrine System", "Hormones", "The Immune System", "Antibodies", "White Blood Cells",
    "Red Blood Cells", "Platelets", "Blood Circulation", "The Heart", "The Lungs",
    "Respiration", "Digestion", "The Liver", "The Kidneys", "The Skin",
    "The Muscular System", "The Skeletal System", "Bones", "Joints", "The Senses",
    "Vision", "Hearing", "Taste", "Smell", "Touch",
    "Memory", "Learning", "Emotion", "Motivation", "Sleep",
    "Dreams", "Consciousness (Neuroscience)", "Language Acquisition", "Child Development", "Aging",
    "Genetic Disorders", "Cancer", "Diabetes", "Obesity", "Mental Health",
    "Depression", "Anxiety", "Schizophrenia", "Autism", "Attention Deficit Disorder"
]


topics = topics + new_topics

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
output_path = "/gpfs/work/vlasenko/07/NN/fatenv/storyteller2/corpus/text"

with open(output_path, "wb") as fp:   #Pickling
    pickle.dump(tlist, fp)



