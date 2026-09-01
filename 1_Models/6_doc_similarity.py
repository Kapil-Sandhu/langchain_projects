from langchain_google_genai import GoogleGenerativeAIEmbeddings # this is used for embedding Models
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

documents = [
    "Launched on December 25, 2021, the James Webb Space Telescope represents a massive leap forward in space astronomy. It was designed to peer deeper into the universe than ever before, observing the formation of the first galaxies.",
    "Unlike the Hubble Space Telescope, which primarily observes in the visible and ultraviolet spectrums, JWST is optimized for infrared light. This allows it to look through dense dust clouds where stars and planetary systems are being born.",
    "To maintain the extreme temperatures required for infrared astronomy, the observatory orbits the second Sun-Earth Lagrange point (L2), located approximately 1 million miles from Earth. This orbit keeps the telescope aligned with the Earth as it moves around the Sun.",
    "The massive, tennis-court-sized sunshield protects the telescope from the heat of the Sun, Earth, and Moon. It consists of five ultra-thin layers of a polyimide film called Kapton, which are specially coated with aluminum for reflectivity and doped-silicon for heat emission.",
    "The telescope's iconic primary mirror is 6.5 meters across and consists of 18 hexagonal segments. These segments are crafted from ultra-lightweight beryllium and coated with a microscopic layer of gold to optimize the reflection of infrared light.",
    "JWST carries four state-of-the-art scientific instruments, including the Near-Infrared Camera (NIRCam) and the Mid-Infrared Instrument (MIRI). These tools are heavily shielded to ensure their detectors remain sensitive enough to capture faint cosmic signals.",
    "While passive cooling is sufficient for most of the telescope, the MIRI instrument requires an active cryocooler system. This specialized refrigerator pumps cold helium gas to bring the instrument's temperature down to a staggering 7 Kelvin.",
    "The deployment sequence of the space observatory was one of the most complex engineering feats ever attempted. Unfolding the massive thermal shield required releasing over 100 pins and carefully tensioning the layers using a system of pulleys and cables.",
    "The geometry of the thermal protection system is critical to its function. By using five distinct layers separated by the vacuum of space, heat is systematically radiated out from between the gaps, ensuring the dark side remains near absolute zero.",
    "Looking ahead, the mission's primary objectives include analyzing the atmospheric composition of potentially habitable exoplanets. By studying the light passing through these alien atmospheres, scientists hope to find biosignatures like water, methane, and carbon dioxide.",
]

quary = "What specific material is the James Webb Space Telescope's sunshield made of?"

emb = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality = 768, # embedding dimesions of gemini model from langchain docs
)

doc_emb = emb.embed_documents(documents)
quary_emb = emb.embed_query(quary)

similary_score = cosine_similarity([quary_emb], doc_emb)[0] 
# we also need to add the positional encoding with the cosine similarty so that why used a 2d list
print(similary_score)

# Pair each score with its original index, and sort from highest to lowest (reverse=True)
sorted_scores = sorted(list(enumerate(similary_score)), key=lambda x: x[1], reverse=True)

print("\n")
print(quary)
print("\nThese 2 are the most relevant paragraphs:\n")

# Loop through the Top 2 most relevent pargraphs in the sorted list
for i in range(2):
    original_index = sorted_scores[i][0]  # The paragraph's position in the original list
    score = sorted_scores[i][1]           # The actual cosine similarity score
    
    print(f"--- Rank {i+1} | Score: {score:.4f} ---")
    print(documents[original_index])
    print()