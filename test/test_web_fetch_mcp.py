# pip install firecrawl-py
from firecrawl import Firecrawl

app = Firecrawl(api_key="fc-1cd6b3753ec04bd38a0285e6d6215ac7")

# Scrape a website:
print(app.scrape('firecrawl.dev'))