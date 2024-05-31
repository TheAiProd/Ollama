```bash
import requests
from bs4 import BeautifulSoup

# Function to scrape data from a webpage
def scrape_webpage(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Modify the following line according to the structure of the webpage you are scraping
    data = soup.find_all('p')  # Example: scraping all paragraphs
    return [p.text for p in data]

url = "https://lilianweng.github.io/"
scraped_data = scrape_webpage(url)

# Count the number of words scraped
word_count = sum(len(paragraph.split()) for paragraph in scraped_data)

print(f"Web scraped: {word_count} words")

# Write output to a file
with open("output.txt", "w") as file:
    for paragraph in scraped_data:
        file.write(paragraph + "\n")
```
