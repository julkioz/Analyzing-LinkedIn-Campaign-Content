import pandas as pd
import re
import emoji

from tfIdfInheritVectorizer.feature_extraction.vectorizer import TFIDFVectorizer
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Load the CSV file. It must be a CSV UTF-8 (Comma delimited)(*.csv) file
# The file was downloaded from Linkedin Campaign Manager > Export > Ad performance.
file_path = "campaign_creative_performance_report.csv"  # Update this to your file path
df = pd.read_csv(file_path, encoding='utf-8', skiprows=5)

# Define helper functions
def calculate_post_length(text):
    return len(text)

def calculate_word_count(text):
    return len(text.split())

def includes_emojis(text):
    return any(char in emoji.EMOJI_DATA for char in text)

def count_hashtags(text):
    return len(re.findall(r"#\w+", text))

def includes_cta(text):
    cta_keywords = ["buy now", "click here", "subscribe", "learn more", "sign up", "get started"]
    return any(keyword in text.lower() for keyword in cta_keywords)

def includes_numbers(text):
    return bool(re.search(r"\d+", text))

def includes_product_description(text):
    product_keywords = ["acrylic", "polyurethane", "pcb materials"]  # Adjust as needed
    return any(keyword in text.lower() for keyword in product_keywords)

def includes_industry_description(text):
    industry_keywords = ["construction", "machinery manufacturing", "utilities", "mining"]  # Adjust as needed
    return any(keyword in text.lower() for keyword in industry_keywords)

# Download stopwords for preprocessing
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean the text and remove stop words
def clean_text(text):
    words = text.lower().split()
    important_words = [word for word in words if word not in stop_words and word.isalpha()]
    return ' '.join(important_words)

# Frequently used word count
def get_frequently_used_words(text):
    word_count = Counter(text.split())
    return word_count.most_common(10)

# Function to calculate keywords using TF-IDF
def extract_keywords_tfidf(text, top_n=10):
    vectorizer = TFIDFVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])
    keywords = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    return [word for word, _ in keywords]

# Apply functions to the DataFrame
df["post_length"] = df["Ad Introduction Text"].apply(calculate_post_length)
df["word_count"] = df["Ad Introduction Text"].apply(calculate_word_count)
df["includes_emojis"] = df["Ad Introduction Text"].apply(includes_emojis)
df["hashtag_count"] = df["Ad Introduction Text"].apply(count_hashtags)
df["includes_cta"] = df["Ad Introduction Text"].apply(includes_cta)
df["includes_numbers"] = df["Ad Introduction Text"].apply(includes_numbers)
df["includes_product_description"] = df["Ad Introduction Text"].apply(includes_product_description)
df["includes_industry_description"] = df["Ad Introduction Text"].apply(includes_industry_description)

# Clean the content by converting it to lowercase, removing stop words, and keeping only relevant words.
df["Ad_Introduction_Text_Tmp"] = df["Ad Introduction Text"].apply(clean_text)
# Frequently used word count
df["Ad_Introduction_Text_Clean"] = df["Ad_Introduction_Text_Tmp"].apply(get_frequently_used_words)
# Keyword Extraction
df["Keyword_Extraction"] = df["Ad_Introduction_Text_Tmp"].apply(extract_keywords_tfidf)

# drop unused columns
df = df.drop(columns=['Ad_Introduction_Text_Tmp', 'Click Through Rate', 'Average CPM', 'Average CPC', 'Reactions', 'Comments', 'Shares', 'Follows', 'Other Clicks', 'Total Social Actions', 'Total Engagements', 'Engagement Rate', 'Viral Impressions', 'Viral Clicks', 'Viral Reactions', 'Viral Comments', 'Viral Shares', 'Viral Follows', 'Viral Other Clicks', 'Conversions', 'Post-Click Conversions', 'View-Through Conversions', 'Conversion Rate', 'Cost per Conversion', 'Total Conversion Value', 'Return on Ad Spend', 'Viral Conversions', 'Viral Post-Click Conversions', 'Viral View-Through Conversions', 'Leads', 'Lead Forms Opened', 'Lead Form Completion Rate', 'Cost per Lead', 'Reach', 'Average Frequency', 'Cost per 1,000 People Reached', 'Event Registrations', 'Click Event Registrations', 'View Event Registrations', 'Viral Event Registrations', 'Viral Click Event Registrations', 'Viral View Event Registrations', 'Clicks to Landing Page', 'Clicks to LinkedIn Page', 'Leads (Work Email)', 'Lead Form Completion Rate (Work Email)', 'Cost Per Lead (Work Email)', 'Member Follows', 'Clicks to Member Profile', 'Qualified Leads', 'Cost Per Qualified Lead', 'Average Dwell Time (in Seconds)', 'Subscriptions', 'Viral Subscriptions'])

# Save the updated DataFrame to a new CSV file
output_file_path = "campaign_creative_performance_report_output.csv"  # Update this if needed
df.to_csv(output_file_path, index=False, encoding='utf-8')

print("Data processing complete. Updated file saved as:", output_file_path)
