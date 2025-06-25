import os
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import scipy.stats as stats
import base64
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import streamlit as st
import google.generativeai as genai
from anthropic import Anthropic
import re
import time
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# Import the necessary module and class for programmatic rerun
import streamlit.runtime.scriptrunner as rst
from streamlit.runtime.scriptrunner import RerunData, RerunException

# Configure page - using only the basic parameters
st.set_page_config(
    page_title="SEO Embedding Analysis Tool",
    layout="wide"
)

# Initialize session state for settings if they don't exist
try:
    if 'google_api_key' not in st.session_state:
        st.session_state.google_api_key = st.secrets.get("GOOGLE_API_KEY", "")
    if 'anthropic_api_key' not in st.session_state:
        st.session_state.anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
except Exception as e:
    # If secrets are not available (local development), initialize with empty strings
    if 'google_api_key' not in st.session_state:
        st.session_state.google_api_key = ""
    if 'anthropic_api_key' not in st.session_state:
        st.session_state.anthropic_api_key = ""

if 'claude_model' not in st.session_state:
    st.session_state.claude_model = "claude-3-7-sonnet-latest"
if 'max_tokens' not in st.session_state:
    st.session_state.max_tokens = 15000
if 'temperature' not in st.session_state:
    st.session_state.temperature = 1.0
if 'thinking_tokens' not in st.session_state:
    st.session_state.thinking_tokens = 8000
if 'embedding' not in st.session_state:
    st.session_state.embedding = None
if 'analysis' not in st.session_state:
    st.session_state.analysis = None
if 'claude_analysis' not in st.session_state:
    st.session_state.claude_analysis = None
if 'content' not in st.session_state:
    st.session_state.content = ""
if 'pdf_data' not in st.session_state:
    st.session_state.pdf_data = None
if 'pdf_generated' not in st.session_state:
    st.session_state.pdf_generated = False
if 'business_type' not in st.session_state:
    st.session_state.business_type = "lead_generation"
if 'page_type' not in st.session_state:
    st.session_state.page_type = "landing_page"
if 'url' not in st.session_state:
    st.session_state.url = ""
if 'fetch_button_clicked' not in st.session_state:
    st.session_state.fetch_button_clicked = False
if 'target_keywords' not in st.session_state:
    st.session_state.target_keywords = ""

# Sidebar for API keys and settings
with st.sidebar:
    st.title("API Settings")

    # API keys - use session state for values
    google_api_key = st.text_input(
        "Google API Key",
        type="password",
        value=st.session_state.google_api_key
    )
    anthropic_api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=st.session_state.anthropic_api_key
    )

    # Model settings
    st.subheader("Model Settings")
    claude_model = st.selectbox(
        "Claude Model",
        ["claude-3-7-sonnet-latest", "claude-3-opus-20240229", "claude-3-5-sonnet-20240620"],
        index=0 if st.session_state.claude_model == "claude-3-7-sonnet-latest" else
              1 if st.session_state.claude_model == "claude-3-opus-20240229" else 2
    )
    max_tokens = st.slider("Max Tokens", 4000, 15000, st.session_state.max_tokens, 1000)
    temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.1)
    thinking_tokens = st.slider("Thinking Tokens", 3000, 8000, st.session_state.thinking_tokens, 1000)

    # Save settings button
    if st.button("Save Settings", type="primary"):
        # Save settings to session state
        st.session_state.google_api_key = google_api_key
        st.session_state.anthropic_api_key = anthropic_api_key
        st.session_state.claude_model = claude_model
        st.session_state.max_tokens = max_tokens
        st.session_state.temperature = temperature
        st.session_state.thinking_tokens = thinking_tokens
        st.success("Settings saved!")

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def get_embedding(text):
    """Get embedding from Google Gemini API"""
    try:
        # Configure API with the provided key
        genai.configure(api_key=st.session_state.google_api_key)

        # Use gemini-embedding-exp-03-07 model as specified
        response = genai.embed_content(
            model="models/gemini-embedding-exp-03-07",
            content=text,
        )
        embedding = response["embedding"]
        return embedding
    except Exception as e:
        st.error(f"Error getting embedding: {e}")
        # Return a random embedding for testing if API fails
        st.warning("Using random embedding instead")
        return np.random.normal(0, 0.1, 3072).tolist()

def get_current_settings():
    return {
        "model": st.session_state.claude_model,
        "max_tokens": st.session_state.max_tokens,
        "temperature": st.session_state.temperature,
        "thinking_tokens": st.session_state.thinking_tokens
    }

def analyze_keyword_presence(text, keywords_list, seo_elements=None):
    """Analyze keyword presence in content and SEO elements"""
    if not keywords_list:
        return {}
    
    text_lower = text.lower()
    keyword_analysis = {}
    
    for keyword in keywords_list:
        keyword_lower = keyword.lower().strip()
        if not keyword_lower:
            continue
            
        # Count occurrences in main content
        content_count = text_lower.count(keyword_lower)
        
        # Analyze keyword presence in SEO elements
        seo_presence = {}
        if seo_elements:
            seo_presence = {
                'title': keyword_lower in seo_elements.get('title', '').lower(),
                'meta_description': keyword_lower in seo_elements.get('meta_description', '').lower(),
                'h1': any(keyword_lower in h1.lower() for h1 in seo_elements.get('h1_tags', [])),
                'h2': any(keyword_lower in h2.lower() for h2 in seo_elements.get('h2_tags', [])),
                'h3': any(keyword_lower in h3.lower() for h3 in seo_elements.get('h3_tags', []))
            }
        
        # Calculate keyword density
        total_words = len(text.split())
        keyword_density = (content_count / total_words * 100) if total_words > 0 else 0
        
        # Determine optimization status
        optimization_status = "not_optimized"
        if content_count > 0:
            if 0.5 <= keyword_density <= 2.5:  # Ideal keyword density range
                optimization_status = "well_optimized"
            elif keyword_density > 2.5:
                optimization_status = "over_optimized"
            else:
                optimization_status = "under_optimized"
        
        keyword_analysis[keyword] = {
            'content_count': content_count,
            'keyword_density': keyword_density,
            'seo_presence': seo_presence,
            'optimization_status': optimization_status,
            'seo_score': sum(seo_presence.values()) if seo_presence else 0
        }
    
    return keyword_analysis

def fetch_content_from_url(url):
    """Fetches and parses content from a given URL with enhanced SEO metadata extraction and keyword analysis."""
    try:
        # Add headers to mimic a real browser request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        
        # Make the request with headers and timeout
        response = requests.get(url, headers=headers, timeout=15)
        
        # Log response details for debugging
        st.write(f"Debug Info: Status Code: {response.status_code}")
        st.write(f"Debug Info: Response Length: {len(response.content)} bytes")
        
        # Check for successful response
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract SEO metadata with enhanced analysis
        title = soup.find('title')
        title_text = title.get_text(strip=True) if title else "No title found"
        
        # Extract meta description
        description = soup.find('meta', attrs={'name': 'description'})
        description_text = description.get('content', '').strip() if description else "No meta description found"
        
        # Extract keywords meta tag
        keywords = soup.find('meta', attrs={'name': 'keywords'})
        keywords_text = keywords.get('content', '').strip() if keywords else "No meta keywords found"
        
        # Extract Open Graph tags for social media
        og_title = soup.find('meta', attrs={'property': 'og:title'})
        og_title_text = og_title.get('content', '').strip() if og_title else "No OG title found"
        
        og_description = soup.find('meta', attrs={'property': 'og:description'})
        og_description_text = og_description.get('content', '').strip() if og_description else "No OG description found"
        
        # Extract all heading tags with enhanced analysis
        h1_tags = soup.find_all('h1')
        h1_texts = [h1.get_text(strip=True) for h1 in h1_tags if h1.get_text(strip=True)]
        
        h2_tags = soup.find_all('h2')
        h2_texts = [h2.get_text(strip=True) for h2 in h2_tags if h2.get_text(strip=True)]
        
        h3_tags = soup.find_all('h3')
        h3_texts = [h3.get_text(strip=True) for h3 in h3_tags if h3.get_text(strip=True)]
        
        h4_tags = soup.find_all('h4')
        h4_texts = [h4.get_text(strip=True) for h4 in h4_tags if h4.get_text(strip=True)]
        
        # Extract alt text from images for accessibility analysis
        img_tags = soup.find_all('img')
        img_alts = [img.get('alt', '').strip() for img in img_tags if img.get('alt', '').strip()]
        missing_alts = len([img for img in img_tags if not img.get('alt', '').strip()])
        
        # Try multiple content extraction strategies
        main_content = None
        
        # Strategy 1: Look for semantic HTML5 tags
        main_content = soup.find('main') or soup.find('article')
        
        # Strategy 2: Look for common content containers
        if not main_content:
            main_content = soup.find('div', class_=lambda x: x and any(
                keyword in x.lower() for keyword in ['content', 'main', 'article', 'post', 'entry']
            ))
        
        # Strategy 3: Look for content by ID
        if not main_content:
            main_content = soup.find('div', id=lambda x: x and any(
                keyword in x.lower() for keyword in ['content', 'main', 'article', 'post']
            ))
        
        # Strategy 4: Fallback to body, but check if it exists
        if not main_content and soup.body:
            main_content = soup.body
        
        # Extract text if content is found
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            st.warning("No main content found, extracting all visible text")
            text = soup.get_text(separator='\n', strip=True)
        
        # Basic cleaning
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Reduce multiple newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Reduce multiple spaces/tabs
        text = text.strip()
        
        # Check if we got meaningful content
        if len(text) < 100:
            st.warning(f"Warning: Only extracted {len(text)} characters. The page might be JavaScript-heavy or protected.")
        
        # Calculate content statistics for SEO analysis
        word_count = len(text.split())
        char_count = len(text)
        
        # Prepare SEO elements for keyword analysis
        seo_elements = {
            'title': title_text,
            'meta_description': description_text,
            'h1_tags': h1_texts,
            'h2_tags': h2_texts,
            'h3_tags': h3_texts
        }
        
        # Analyze target keywords if provided
        keywords_list = []
        keyword_analysis = {}
        if st.session_state.target_keywords.strip():
            keywords_list = [kw.strip() for kw in st.session_state.target_keywords.split('\n') if kw.strip()]
            keyword_analysis = analyze_keyword_presence(text, keywords_list, seo_elements)
        
        # Build comprehensive extracted information with enhanced SEO metadata and keyword analysis
        extracted_info = "=== SEO METADATA ANALYSIS ===\n\n"
        
        # Title analysis
        extracted_info += f"TITLE TAG:\n"
        extracted_info += f"Content: {title_text}\n"
        extracted_info += f"Length: {len(title_text)} characters\n"
        extracted_info += f"SEO Status: {'Good length' if 30 <= len(title_text) <= 60 else 'Consider optimizing (30-60 chars recommended)'}\n\n"
        
        # Meta description analysis
        extracted_info += f"META DESCRIPTION:\n"
        extracted_info += f"Content: {description_text}\n"
        extracted_info += f"Length: {len(description_text)} characters\n"
        extracted_info += f"SEO Status: {'Good length' if 120 <= len(description_text) <= 160 else 'Consider optimizing (120-160 chars recommended)'}\n\n"
        
        # Keywords analysis
        if keywords_text != "No meta keywords found":
            extracted_info += f"META KEYWORDS: {keywords_text}\n\n"
        
        # Open Graph analysis
        if og_title_text != "No OG title found" or og_description_text != "No OG description found":
            extracted_info += "OPEN GRAPH TAGS:\n"
            extracted_info += f"OG Title: {og_title_text}\n"
            extracted_info += f"OG Description: {og_description_text}\n\n"
        
        # Heading structure analysis
        extracted_info += "HEADING STRUCTURE ANALYSIS:\n"
        
        if h1_texts:
            extracted_info += f"H1 Tags ({len(h1_texts)} found):\n"
            for i, h1 in enumerate(h1_texts, 1):
                extracted_info += f"  {i}. {h1} (Length: {len(h1)} chars)\n"
            extracted_info += f"H1 Analysis: {'Single H1 found' if len(h1_texts) == 1 else 'Multiple H1s detected - consider using only one'}\n\n"
        else:
            extracted_info += "H1 Tags: No H1 tags found - Critical SEO issue!\n\n"
        
        if h2_texts:
            extracted_info += f"H2 Tags ({len(h2_texts)} found):\n"
            for i, h2 in enumerate(h2_texts[:8], 1):  # Show first 8 H2s
                extracted_info += f"  {i}. {h2}\n"
            if len(h2_texts) > 8:
                extracted_info += f"  ... and {len(h2_texts) - 8} more H2 tags\n"
            extracted_info += "\n"
        
        if h3_texts:
            extracted_info += f"H3 Tags ({len(h3_texts)} found):\n"
            for i, h3 in enumerate(h3_texts[:6], 1):  # Show first 6 H3s
                extracted_info += f"  {i}. {h3}\n"
            if len(h3_texts) > 6:
                extracted_info += f"  ... and {len(h3_texts) - 6} more H3 tags\n"
            extracted_info += "\n"
        
        if h4_texts:
            extracted_info += f"H4 Tags: {len(h4_texts)} found\n\n"
        
        # Keyword Analysis Section
        if keyword_analysis:
            extracted_info += "=== KEYWORD TARGETING ANALYSIS ===\n\n"
            
            for keyword, analysis in keyword_analysis.items():
                extracted_info += f"KEYWORD: {keyword}\n"
                extracted_info += f"Content Mentions: {analysis['content_count']}\n"
                extracted_info += f"Keyword Density: {analysis['keyword_density']:.2f}%\n"
                extracted_info += f"Optimization Status: {analysis['optimization_status'].replace('_', ' ').title()}\n"
                
                # SEO element presence
                seo_presence = analysis['seo_presence']
                extracted_info += f"SEO Element Presence:\n"
                extracted_info += f"  • Title Tag: {'Yes' if seo_presence.get('title') else 'No'}\n"
                extracted_info += f"  • Meta Description: {'Yes' if seo_presence.get('meta_description') else 'No'}\n"
                extracted_info += f"  • H1 Tags: {'Yes' if seo_presence.get('h1') else 'No'}\n"
                extracted_info += f"  • H2 Tags: {'Yes' if seo_presence.get('h2') else 'No'}\n"
                extracted_info += f"  • H3 Tags: {'Yes' if seo_presence.get('h3') else 'No'}\n"
                extracted_info += f"SEO Integration Score: {analysis['seo_score']}/5\n\n"
        
        # Content statistics
        extracted_info += "=== CONTENT STATISTICS ===\n"
        extracted_info += f"Word Count: {word_count}\n"
        extracted_info += f"Character Count: {char_count}\n"
        extracted_info += f"Content Length Assessment: {'Good length' if word_count >= 300 else 'Consider adding more content (300+ words recommended)'}\n\n"
        
        # Image SEO analysis
        if img_tags:
            extracted_info += f"IMAGE SEO ANALYSIS:\n"
            extracted_info += f"Total Images: {len(img_tags)}\n"
            extracted_info += f"Images with Alt Text: {len(img_alts)}\n"
            extracted_info += f"Missing Alt Text: {missing_alts}\n"
            extracted_info += f"Alt Text Status: {'All images have alt text' if missing_alts == 0 else f'{missing_alts} images missing alt text'}\n\n"
        
        extracted_info += "=== PAGE CONTENT ===\n\n"
        extracted_info += text
        
        # Display enhanced SEO metadata summary in Streamlit with keyword analysis
        st.success("Enhanced SEO Analysis with Keyword Targeting completed successfully!")
        
        with st.expander("SEO & Keyword Analysis Summary", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Title Tag Analysis**")
                st.write(f"• Content: {title_text}")
                st.write(f"• Length: {len(title_text)} characters")
                if 30 <= len(title_text) <= 60:
                    st.success("Good length")
                else:
                    st.warning("Consider optimizing (30-60 chars)")
                
                st.markdown("**Meta Description Analysis**")
                st.write(f"• Content: {description_text}")
                st.write(f"• Length: {len(description_text)} characters")
                if 120 <= len(description_text) <= 160:
                    st.success("Good length")
                else:
                    st.warning("Consider optimizing (120-160 chars)")
            
            with col2:
                st.markdown("**Heading Structure**")
                if h1_texts:
                    if len(h1_texts) == 1:
                        st.success(f"Single H1: {h1_texts[0]}")
                    else:
                        st.warning(f"{len(h1_texts)} H1 tags found")
                        for i, h1 in enumerate(h1_texts, 1):
                            st.write(f"  {i}. {h1}")
                else:
                    st.error("No H1 tags found")
                
                st.write(f"• H2 Tags: {len(h2_texts)}")
                st.write(f"• H3 Tags: {len(h3_texts)}")
                
                st.markdown("**Content Analysis**")
                st.write(f"• Word Count: {word_count}")
                if word_count >= 300:
                    st.success("Good content length")
                else:
                    st.warning("Consider adding more content")
            
            with col3:
                if keyword_analysis:
                    st.markdown("**Keyword Analysis**")
                    st.write(f"• Keywords Analyzed: {len(keyword_analysis)}")
                    
                    # Show top 3 keywords with best optimization
                    sorted_keywords = sorted(keyword_analysis.items(), 
                                           key=lambda x: (x[1]['seo_score'], x[1]['content_count']), 
                                           reverse=True)
                    
                    for keyword, analysis in sorted_keywords[:3]:
                        status_icon = {
                            'well_optimized': 'Good',
                            'under_optimized': 'Needs Work',
                            'over_optimized': 'Over-Optimized',
                            'not_optimized': 'Not Optimized'
                        }.get(analysis['optimization_status'], 'Not Optimized')
                        
                        st.write(f"**{keyword}**")
                        st.write(f"  Status: {status_icon}")
                        st.write(f"  Density: {analysis['keyword_density']:.1f}% | SEO Score: {analysis['seo_score']}/5")
                else:
                    st.info("Add target keywords for detailed analysis")
        
        return extracted_info
        
    except requests.exceptions.Timeout:
        st.error("Request timed out. The website might be slow or unresponsive.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Connection error. Check your internet connection or the URL.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error {e.response.status_code}: {e.response.reason}")
        if e.response.status_code == 403:
            st.error("Access forbidden. The website might be blocking automated requests.")
        elif e.response.status_code == 404:
            st.error("Page not found. Please check the URL.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request error: {e}")
        return None
    except Exception as e:
        st.error(f"Error parsing content: {e}")
        return None

def analyze_with_claude(embedding_data, content_snippet, business_type, page_type):
    """Get analysis from Claude with business, page type, and keyword targeting context"""
    try:
        # Get current settings at the time of function call
        current_settings = get_current_settings()

        # Initialize Anthropic client with the provided key
        anthropic_client = Anthropic(api_key=st.session_state.anthropic_api_key)

        # Create the business type context string
        business_context = ""
        if business_type == "lead_generation":
            business_context = "a lead generation or service-based business focused on converting visitors to leads or clients"
        elif business_type == "ecommerce":
            business_context = "an e-commerce business focused on selling products online and maximizing conversions"
        elif business_type == "saas":
            business_context = "a SaaS or technology company focused on showcasing features and driving sign-ups"
        elif business_type == "educational":
            business_context = "an educational platform or information resource focused on providing valuable content"
        else:  # local_business
            business_context = "a local business focused on driving local customers and in-person visits"

        # Create the page type context string
        page_context = ""
        if page_type == "landing_page":
            page_context = "a landing page designed to convert visitors for a specific purpose"
        elif page_type in ["service_page", "product_page", "feature_page"]:
            page_context = f"a {page_type.replace('_', ' ')} meant to showcase specific offerings and drive interest"
        elif page_type == "blog_post":
            page_context = "a blog post aimed at providing informational content and building authority"
        elif page_type == "homepage":
            page_context = "a homepage that serves as the main entry point to the website"
        elif page_type in ["about_us", "contact_page"]:
            page_context = f"an {page_type.replace('_', ' ')} providing company information and building trust"
        elif page_type == "category_page":
            page_context = "a category page listing multiple products with filtering options"
        elif page_type == "checkout_page":
            page_context = "a checkout page designed to complete transactions smoothly"
        elif page_type == "pricing_page":
            page_context = "a pricing page designed to showcase different plans and drive conversions"
        elif page_type == "documentation":
            page_context = "documentation content aimed at helping existing users"
        elif page_type == "course_page":
            page_context = "a course page designed to showcase educational offerings"
        elif page_type == "resource_page":
            page_context = "a resource page providing valuable downloadable content"
        else:
            page_context = f"a {page_type.replace('_', ' ')}"

        # Process target keywords
        keywords_context = ""
        keywords_list = []
        if st.session_state.target_keywords.strip():
            keywords_list = [kw.strip() for kw in st.session_state.target_keywords.split('\n') if kw.strip()]
            keywords_context = f"The target keywords for this content are: {', '.join(keywords_list[:10])}" # Limit to first 10 for context
            if len(keywords_list) > 10:
                keywords_context += f" and {len(keywords_list) - 10} more keywords."
        else:
            keywords_context = "No specific target keywords have been provided."

        message = anthropic_client.messages.create(
            model=current_settings["model"],
            max_tokens=current_settings["max_tokens"],
            temperature=current_settings["temperature"],
            thinking={
                "type": "enabled",
                "budget_tokens": current_settings["thinking_tokens"]
            },
            system=f"""You are an advanced SEO and NLP Embedding Analysis Expert with deep expertise in semantic content optimization, machine learning-driven content strategy, keyword targeting analysis, and advanced natural language processing techniques.

Your mission is to provide a comprehensive, multi-dimensional analysis of embedding data that transforms raw numerical information into actionable SEO and content strategy insights, with a special focus on keyword optimization.

IMPORTANT CONTEXT: 
- The content being analyzed is for {business_context}
- Specifically, it is {page_context}
- {keywords_context}
- Tailor all your analysis and recommendations to this specific business type, page type, and keyword targeting strategy

## ENHANCED ANALYSIS METHODOLOGY WITH KEYWORD TARGETING

1. **Keyword-Semantic Alignment Analysis**:
   - Identify embedding dimensions that correlate with target keyword themes
   - Analyze semantic gaps between target keywords and actual content signals
   - Evaluate keyword integration quality through embedding density patterns
   - Assess keyword cannibalization risks and opportunities
   - Identify long-tail keyword opportunities based on semantic clustering

2. **SEO Metadata & Keyword Integration Analysis**:
   - Analyze the structured SEO metadata for keyword integration effectiveness
   - Correlate SEO elements with target keyword semantic signals in embeddings
   - Identify gaps between SEO promises and keyword-focused content delivery
   - Evaluate keyword placement strategy across title, description, and headings
   - Assess keyword density optimization through semantic signal strength

3. **Dimension Analysis Method with Keyword Focus**:
   - First identify the top 20 dimensions by absolute activation magnitude
   - Cluster these dimensions into 3-7 related groups based on activation patterns
   - Cross-reference dimension clusters with target keyword themes and semantic fields
   - Identify which keywords are well-represented vs. under-represented in embedding space
   - Interpret semantic clusters in context of keyword targeting strategy
   - Never assign predetermined meanings to specific dimension ranges
   - Base all interpretations on patterns present in current embedding, SEO data, and keyword targets

4. **Keyword Optimization Opportunity Analysis**:
   - Identify underutilized semantic spaces that could support target keywords
   - Analyze competitor keyword gaps through semantic density analysis
   - Evaluate keyword difficulty based on semantic signal strength
   - Assess content topical authority for each target keyword theme
   - Identify related keyword opportunities through embedding clustering

5. **Statistical Consistency with Keyword Metrics**:
   - Always flag dimensions with magnitude >0.1 as significant for keyword analysis
   - Consider clusters of 3+ adjacent dimensions as potential keyword theme areas
   - Identify keyword semantic gaps as dimension ranges with low activation (<0.01)
   - Use standard deviation to measure keyword theme distribution across content
   - Provide specific dimension numbers when referencing keyword-related activation patterns

## CONTEXTUAL EXPLANATION
Begin with a comprehensive explanation integrating embedding analysis with keyword targeting:

**Embedding Dimensions & Keyword Targeting Explained:**
- Explain how embedding dimensions represent semantic features that directly relate to keyword themes
- Clarify how target keywords should activate specific dimension clusters when well-optimized
- Note how keyword semantic signals appear as consistent activation patterns across related dimensions
- Explain how keyword optimization gaps show up as weak or missing activation in relevant semantic areas
- Give concrete examples of how dimension clusters relate to specific keyword optimization opportunities

**Keyword-Semantic Integration Analysis:**
- Explain how well-targeted keywords create strong, consistent embedding signals
- Describe how keyword density correlates with embedding activation strength
- Clarify how semantic keyword relationships appear in dimension clustering patterns
- Note how topical authority for keywords manifests as broad, consistent activation across related dimensions
- Explain how keyword cannibalization appears as competing or conflicting semantic signals

## ANALYSIS STRUCTURE
Your output must follow this structure precisely with SIX distinct sections:

1. **Contextual Explanation**
   - Provide explanation of embedding dimensions, SEO metadata, and keyword targeting integration
   - Use accessible language while maintaining technical accuracy
   - Connect abstract concepts to practical SEO and keyword optimization implications

2. **SEO Metadata Analysis**
   - Analyze the extracted SEO metadata (title, description, headings) for optimization potential
   - Evaluate SEO element quality, length, and effectiveness
   - Identify SEO metadata strengths and improvement opportunities
   - Assess alignment between different SEO elements

3. **Keyword Targeting Analysis**
   - Analyze each target keyword's semantic representation in the embedding
   - Evaluate keyword integration across SEO elements and content
   - Identify keyword optimization opportunities and gaps
   - Assess keyword density and distribution patterns
   - Evaluate keyword difficulty and competitiveness based on semantic signals

4. **Embedding-SEO-Keyword Correlation Analysis**
   - Analyze how embedding patterns correlate with both SEO metadata and target keywords
   - Identify semantic alignment or misalignment between keywords, SEO elements, and content reality
   - Evaluate topical authority signals for each target keyword theme
   - Assess content semantic density relative to keyword targeting goals
   - Identify content gaps that could be filled to better support target keywords

5. **Actionable Recommendations**
   - Provide specific, implementable suggestions that combine SEO best practices with keyword targeting and embedding insights
   - Separate completely from analysis sections
   - Include examples of how to implement each recommendation
   - Focus on recommendations that leverage SEO metadata optimization, keyword targeting, and semantic content enhancement
   - Prioritize recommendations by impact on target keyword rankings
   - Tailor specifically for {business_context} and {page_context}

6. **Summary**
   - Summarize key findings from SEO, embedding, and keyword analysis in non-technical language
   - Format as bullet points combining SEO, semantic, and keyword insights
   - Include specific keyword opportunities and quick wins
   - Ensure it's understandable by someone with no technical background

## BUSINESS-SPECIFIC CONSIDERATIONS

For {business_context}:
- Focus on keyword strategies that align with this business model's conversion funnel
- Consider the typical customer journey and keyword search intent for this business type
- Align keyword recommendations with business goals and revenue drivers
- Emphasize keywords that matter most for this business type's success metrics

For {page_context}:
- Consider keywords that align with this page type's specific purpose
- Focus on keyword optimization that enhances the primary goal of this page type
- Address keyword search intent that matches this page type's user expectations
- Optimize for keywords that drive the right kind of traffic for this page type

## KEYWORD-SPECIFIC ANALYSIS REQUIREMENTS

When target keywords are provided:
- Analyze each keyword individually for semantic representation strength
- Evaluate keyword competition level based on semantic density requirements
- Identify semantic keyword clusters and related term opportunities
- Assess keyword cannibalization risks between multiple targets
- Provide specific keyword density recommendations (aim for 0.5-2.5% for primary keywords)
- Suggest keyword placement priorities (title > H1 > H2 > meta description > content)

When no target keywords are provided:
- Identify implied keyword targets based on content semantic analysis
- Suggest keyword opportunities based on strong semantic signals in embedding
- Recommend keyword research directions based on semantic clustering patterns
- Highlight content themes that could be developed into keyword targets""",
            messages=[
                {
                    "role": "user",
                    "content": f"""Comprehensive SEO-Enhanced Embedding Analysis with Keyword Targeting:

CONTENT WITH SEO METADATA AND KEYWORD ANALYSIS:
{content_snippet[:7000]}

TARGET KEYWORDS ANALYSIS:
{f"Target Keywords: {st.session_state.target_keywords}" if st.session_state.target_keywords.strip() else "No target keywords specified - please analyze content for keyword opportunities"}

EMBEDDING DATA DIAGNOSTICS:
- Total Dimensions: {len(embedding_data)}
- Dimensional Statistics:
  * Mean Value: {np.mean(embedding_data):.6f}
  * Standard Deviation: {np.std(embedding_data):.6f}
  * Minimum Value: {np.min(embedding_data):.6f} (Dimension {np.argmin(embedding_data)})
  * Maximum Value: {np.max(embedding_data):.6f} (Dimension {np.argmax(embedding_data)})

TOP ACTIVATION DIMENSIONS:
{sorted(range(len(embedding_data)), key=lambda i: abs(embedding_data[i]), reverse=True)[:15]}

BUSINESS TYPE: {business_type}
PAGE TYPE: {page_type}

Please provide a comprehensive analysis following the exact format in your instructions:
1. CONTEXTUAL EXPLANATION of embedding dimensions, SEO metadata, and keyword targeting integration
2. SEO METADATA ANALYSIS of the extracted SEO elements
3. KEYWORD TARGETING ANALYSIS with specific evaluation of each target keyword
4. EMBEDDING-SEO-KEYWORD CORRELATION ANALYSIS showing how all elements align
5. ACTIONABLE RECOMMENDATIONS combining SEO, keyword, and embedding insights with priority ranking
6. PLAIN LANGUAGE SUMMARY including specific keyword opportunities and quick wins

Ensure your analysis transforms embedding data, SEO metadata, and keyword targeting into a cohesive, strategic optimization plan."""
                }
            ]
        )

        # Extract text content from Claude's response
        if hasattr(message.content, '__iter__') and not isinstance(message.content, str):
            # If content is an iterable (like a list of content blocks)
            extracted_text = ""
            for block in message.content:
                if hasattr(block, 'text'):
                    extracted_text += block.text
                elif isinstance(block, str):
                    extracted_text += block
            return extracted_text
        elif hasattr(message.content, 'text'):
            # If content is a single TextBlock object
            return message.content.text
        else:
            # If content is already a string or something else
            return str(message.content)

    except Exception as e:
        st.error(f"Error getting Claude analysis: {e}")
        return "Error getting analysis from Claude. Please check your API key and try again."

def format_claude_analysis(analysis_text):
    # Convert markdown headings (##, ###) into <strong> tags
    formatted_text = re.sub(r'^###?\s*(.+)

def plot_embedding_overview(embedding):
    """Create overview plot of embedding values"""
    try:
        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        ax.plot(range(len(embedding)), embedding)
        ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax.set_title('Embedding Values Across All 3k Dimensions')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        return fig
    except Exception as e:
        st.error(f"Error plotting embedding overview: {e}")
        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center')
        return fig

def plot_top_dimensions(embedding):
    """Plot top dimensions by magnitude"""
    try:
        # Get indices of top 20 dimensions by magnitude
        top_indices = sorted(range(len(embedding)), key=lambda i: abs(embedding[i]), reverse=True)[:20]
        top_values = [embedding[i] for i in top_indices]

        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        colors = ['blue' if v >= 0 else 'red' for v in top_values]
        ax.bar(range(len(top_indices)), top_values, color=colors)
        ax.set_xticks(range(len(top_indices)))
        ax.set_xticklabels(top_indices, rotation=45)
        ax.set_title('Top 20 Dimensions by Magnitude')
        ax.set_xlabel('Dimension Index')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        return fig
    except Exception as e:
        st.error(f"Error plotting top dimensions: {e}")
        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center')
        return fig

def plot_dimension_clusters(embedding):
    """Plot dimension clusters heatmap"""
    try:
        # Reshape embedding to highlight patterns
        embedding_reshaped = np.array(embedding).reshape(64, 48)

        fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
        # Create a custom colormap from blue to white to red
        cmap = LinearSegmentedColormap.from_list('BrBG', ['blue', 'white', 'red'], N=256)
        im = ax.imshow(embedding_reshaped, cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax, label='Activation Value')
        ax.set_title('Embedding Clusters Heatmap (Reshaped to 64x48)')
        ax.set_xlabel('Dimension Group')
        ax.set_ylabel('Dimension Group')
        return fig
    except Exception as e:
        st.error(f"Error plotting dimension clusters: {e}")
        fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center')
        return fig

def plot_pca(embedding):
    """Plot PCA visualization of embedding dimensions"""
    try:
        # Create a 2D array where each row is a segment of the original embedding
        segment_size = 256
        num_segments = len(embedding) // segment_size
        data_matrix = np.zeros((num_segments, segment_size))

        # Fill the matrix with segments
        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size
            data_matrix[i] = embedding[start:end]

        # Apply PCA
        fig = plt.figure(figsize=(10, 8))
        if num_segments > 1:
            pca = PCA(n_components=2)
            pca_results = pca.fit_transform(data_matrix)

            plt.scatter(pca_results[:, 0], pca_results[:, 1])

            # Label each point with its segment range
            for i in range(num_segments):
                start = i * segment_size
                end = start + segment_size - 1
                plt.annotate(f"{start}-{end}",
                            (pca_results[i, 0], pca_results[i, 1]),
                            fontsize=8)

            plt.title('PCA of Embedding Segments')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.grid(True, alpha=0.3)
        else:
            # If we don't have enough segments, create a simpler visualization
            plt.text(0.5, 0.5, "Not enough segments for PCA visualization",
                    ha='center', va='center', fontsize=12)
            plt.axis('off')

        return fig
    except Exception as e:
        st.error(f"Error plotting PCA: {e}")
        fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center')
        return fig

def plot_activation_histogram(embedding):
    """Plot histogram of embedding activation values"""
    try:
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        ax.hist(embedding, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        ax.set_title('Distribution of Embedding Values')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        return fig
    except Exception as e:
        st.error(f"Error plotting activation histogram: {e}")
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center')
        return fig

def analyze_embedding(embedding):
    """Analyze embedding for key metrics"""
    try:
        embedding = np.array(embedding)  # Convert to numpy array for easier processing
        abs_embedding = np.abs(embedding)

        # Calculate key metrics - CONVERT NUMPY TYPES TO PYTHON NATIVE TYPES
        metrics = {
            "dimension_count": int(len(embedding)),
            "mean_value": float(np.mean(embedding)),
            "std_dev": float(np.std(embedding)),
            "min_value": float(np.min(embedding)),
            "min_dimension": int(np.argmin(embedding)),
            "max_value": float(np.max(embedding)),
            "max_dimension": int(np.argmax(embedding)),
            "median_value": float(np.median(embedding)),
            "positive_count": int(np.sum(embedding > 0)),
            "negative_count": int(np.sum(embedding < 0)),
            "zero_count": int(np.sum(embedding == 0)),
            "abs_mean": float(np.mean(abs_embedding)),
            "significant_dims": int(np.sum(abs_embedding > 0.1))
        }

        # Find activation clusters
        significant_threshold = 0.1
        significant_dims = np.where(abs_embedding > significant_threshold)[0]

        # Find clusters (dimensions that are close to each other)
        clusters = []
        if len(significant_dims) > 0:
            current_cluster = [int(significant_dims[0])]  # Convert to int

            for i in range(1, len(significant_dims)):
                if significant_dims[i] - significant_dims[i-1] <= 5:  # If dimensions are close
                    current_cluster.append(int(significant_dims[i]))  # Convert to int
                else:
                    if len(current_cluster) > 0:
                        clusters.append(current_cluster)
                    current_cluster = [int(significant_dims[i])]  # Convert to int

            if len(current_cluster) > 0:
                clusters.append(current_cluster)

        # Filter to meaningful clusters (more than 1 dimension)
        clusters = [c for c in clusters if len(c) > 1]

        # Format clusters for display
        cluster_info = []
        for i, cluster in enumerate(clusters):
            values = [float(embedding[dim]) for dim in cluster]  # Convert to float
            cluster_info.append({
                "id": i+1,
                "dimensions": [int(d) for d in cluster],  # Convert to int
                "start_dim": int(min(cluster)),
                "end_dim": int(max(cluster)),
                "size": int(len(cluster)),
                "avg_value": float(np.mean(values)),
                "max_value": float(np.max(values)),
                "max_dim": int(cluster[np.argmax(values)])
            })

        # Top dimensions by magnitude
        top_indices = sorted(range(len(embedding)), key=lambda i: abs(embedding[i]), reverse=True)[:10]
        top_dimensions = [{"dimension": int(idx), "value": float(embedding[idx])} for idx in top_indices]

        return {
            "metrics": metrics,
            "clusters": cluster_info,
            "top_dimensions": top_dimensions
        }
    except Exception as e:
        st.error(f"Error analyzing embedding: {e}")
        # Return a minimal valid structure in case of error
        return {
            "metrics": {
                "dimension_count": 0,
                "mean_value": 0.0,
                "std_dev": 0.0,
                "min_value": 0.0,
                "min_dimension": 0,
                "max_value": 0.0,
                "max_dimension": 0,
                "median_value": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "zero_count": 0,
                "abs_mean": 0.0,
                "significant_dims": 0
            },
            "clusters": [],
            "top_dimensions": []
        }

def create_report_pdf(embedding, analysis, claude_analysis, business_type, page_type):
    """Create a better-formatted PDF report of the analysis results with business and page type context"""
    try:
        # Create a buffer for the PDF
        buffer = BytesIO()

        # Create the PDF document with better margins
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )

        # Create custom styles
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading1_style = styles['Heading1']
        heading2_style = styles['Heading2']

        # Create a better normal style with more space
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            spaceBefore=6,
            spaceAfter=6,
            leading=14  # Line spacing
        )

        # Create a story (content)
        story = []

        # Add title with better spacing
        story.append(Paragraph("SEO Embedding Analysis Report", title_style))
        story.append(Spacer(1, 24))  # More space after title

        # Add date with better formatting
        from datetime import datetime
        date_style = ParagraphStyle(
            'DateStyle',
            parent=normal_style,
            fontName='Helvetica-Oblique',
            textColor=colors.darkblue
        )
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", date_style))
        story.append(Spacer(1, 12))  # Space after date

        # Add content categorization information
        category_style = ParagraphStyle(
            'CategoryStyle',
            parent=normal_style,
            fontName='Helvetica-Bold',
            textColor=colors.darkblue,
            fontSize=11
        )

        # Convert internal values to display values
        business_display = {
            "lead_generation": "Lead Generation/Service",
            "ecommerce": "E-commerce",
            "saas": "SaaS/Tech",
            "educational": "Educational/Informational",
            "local_business": "Local Business"
        }.get(business_type, business_type.replace("_", " ").title())

        page_display = page_type.replace("_", " ").title()

        story.append(Paragraph(f"Business Type: {business_display}", category_style))
        story.append(Paragraph(f"Page Type: {page_display}", category_style))
        
        # Add keyword information if available
        if st.session_state.target_keywords.strip():
            keywords_list = [kw.strip() for kw in st.session_state.target_keywords.split('\n') if kw.strip()]
            keywords_text = ", ".join(keywords_list[:5])
            if len(keywords_list) > 5:
                keywords_text += f" (+{len(keywords_list) - 5} more)"
            story.append(Paragraph(f"Target Keywords: {keywords_text}", category_style))
        
        story.append(Spacer(1, 24))  # More space after categories

        # Add metrics section with better formatting
        story.append(Paragraph("Key Metrics", heading1_style))
        story.append(Spacer(1, 12))

        # Format metrics as a table for better appearance
        metrics = analysis["metrics"]
        metrics_data = [
            ["Metric", "Value"],
            ["Dimensions", f"{metrics['dimension_count']}"],
            ["Mean Value", f"{metrics['mean_value']:.6f}"],
            ["Standard Deviation", f"{metrics['std_dev']:.6f}"],
            ["Min Value", f"{metrics['min_value']:.6f} (dim {metrics['min_dimension']})"],
            ["Max Value", f"{metrics['max_value']:.6f} (dim {metrics['max_dimension']})"],
            ["Positive Values", f"{metrics['positive_count']} ({metrics['positive_count']/metrics['dimension_count']*100:.2f}%)"],
            ["Negative Values", f"{metrics['negative_count']} ({metrics['negative_count']/metrics['dimension_count']*100:.2f}%)"],
            ["Significant Dimensions", f"{metrics['significant_dims']} (>0.1)"]
        ]

        # Create a table with the metrics
        metrics_table = Table(metrics_data, colWidths=[2*inch, 3.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 24))

        # Create visualizations page with better layout
        story.append(PageBreak())
        story.append(Paragraph("Embedding Visualizations", heading1_style))
        story.append(Spacer(1, 16))

        # Function to add an image with a better caption
        def add_figure_with_caption(story, img_data, caption):
            # Add caption with better styling
            caption_style = ParagraphStyle(
                'CaptionStyle',
                parent=heading2_style,
                textColor=colors.darkblue,
                spaceAfter=8
            )
            story.append(Paragraph(caption, caption_style))

            # Create image with better size settings
            img = Image(img_data, width=7*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 16))

        # Create plots with better resolution and save them as images
        # 1. Embedding Overview
        fig_overview = plot_embedding_overview(embedding)
        fig_overview.set_dpi(150)  # Higher resolution
        fig_overview.tight_layout()  # Better layout
        overview_img = BytesIO()
        fig_overview.savefig(overview_img, format='png', bbox_inches='tight')
        plt.close(fig_overview)
        overview_img.seek(0)

        # 2. Top Dimensions
        fig_top = plot_top_dimensions(embedding)
        fig_top.set_dpi(150)
        fig_top.tight_layout()
        top_img = BytesIO()
        fig_top.savefig(top_img, format='png', bbox_inches='tight')
        plt.close(fig_top)
        top_img.seek(0)

        # 3. Activation Histogram
        fig_hist = plot_activation_histogram(embedding)
        fig_hist.set_dpi(150)
        fig_hist.tight_layout()
        hist_img = BytesIO()
        fig_hist.savefig(hist_img, format='png', bbox_inches='tight')
        plt.close(fig_hist)
        hist_img.seek(0)

        # Add the first three visualizations
        add_figure_with_caption(story, overview_img, "Embedding Overview")
        add_figure_with_caption(story, top_img, "Top 20 Dimensions by Magnitude")
        add_figure_with_caption(story, hist_img, "Distribution of Embedding Values")

        # Start a new page for the next visualizations
        story.append(PageBreak())

        # 4. Dimension Clusters
        fig_clusters = plot_dimension_clusters(embedding)
        fig_clusters.set_dpi(150)
        fig_clusters.tight_layout()
        clusters_img = BytesIO()
        fig_clusters.savefig(clusters_img, format='png', bbox_inches='tight')
        plt.close(fig_clusters)
        clusters_img.seek(0)

        # 5. PCA Visualization
        fig_pca = plot_pca(embedding)
        fig_pca.set_dpi(150)
        fig_pca.tight_layout()
        pca_img = BytesIO()
        fig_pca.savefig(pca_img, format='png', bbox_inches='tight')
        plt.close(fig_pca)
        pca_img.seek(0)

        # Add the remaining visualizations
        add_figure_with_caption(story, clusters_img, "Dimension Clusters Heatmap")
        add_figure_with_caption(story, pca_img, "PCA Visualization of Embedding Segments")

        # Add Claude Analysis section with better formatting
        story.append(PageBreak())
        story.append(Paragraph("Comprehensive Analysis", heading1_style))
        story.append(Spacer(1, 16))

        # Add categorization banner for context
        context_style = ParagraphStyle(
            'ContextStyle',
            parent=normal_style,
            fontName='Helvetica-Bold',
            textColor=colors.white,
            fontSize=10,
            alignment=1,  # Center alignment
            backColor=colors.darkblue
        )

        context_text = f"Analysis tailored for: {business_display} | {page_display}"
        if st.session_state.target_keywords.strip():
            keywords_list = [kw.strip() for kw in st.session_state.target_keywords.split('\n') if kw.strip()]
            context_text += f" | Keywords: {len(keywords_list)} targeted"

        story.append(Paragraph(context_text, context_style))
        story.append(Spacer(1, 12))

        # Process Claude analysis text - convert from HTML/markdown to plain text
        # Remove HTML tags
        analysis_text = re.sub(r'<[^>]*>', '', claude_analysis)

        # Function to process headings better
        def process_heading(text, style):
            text = text.strip()
            if text.startswith('##'):
                return Paragraph(text.replace('##', '').strip(), heading1_style)
            elif text.startswith('#'):
                return Paragraph(text.replace('#', '').strip(), heading2_style)
            elif text.strip() and text.strip()[0].isupper():  # Likely a section title without # marks
                first_line = text.split("\n")[0].strip()
                if len(first_line) < 50 and first_line.isupper():  # It's probably a heading
                    return Paragraph(first_line, heading2_style)
            # Normal paragraph
            return Paragraph(text, normal_style)

        # Split by paragraphs and process each one
        paragraphs = analysis_text.split('\n\n')

        for paragraph in paragraphs:
            if paragraph.strip():
                element = process_heading(paragraph, normal_style)
                story.append(element)
                if isinstance(element, Paragraph) and element.style == heading1_style:
                    story.append(Spacer(1, 12))  # More space after main headings

        # Build the PDF
        doc.build(story)

        # Get the PDF value
        pdf_value = buffer.getvalue()
        buffer.close()

        return pdf_value

    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

# Main content area
def main():
    st.title("SEO Embedding Analysis Tool")

    # Content input options
    st.header("Content Input")

    input_method = st.radio("Choose input method:", ("Paste Content", "Fetch from URL"), key="input_method_radio")

    # Use a placeholder for content that reflects the current input method
    content_placeholder = "Enter the content you want to analyze..." if input_method == "Paste Content" else "Content will be fetched from the URL..."

    if input_method == "Paste Content":
        # Text area for pasting content
        # Value is read directly from session state on rerun
        st.session_state.content = st.text_area("Paste your content here:", height=300,
                              placeholder=content_placeholder,
                              value=st.session_state.get('content', ''), key="pasted_content_area")
        st.session_state.url = "" # Clear URL state when pasting content
        # Button for analyzing pasted content
        analyze_pasted_button = st.button("Analyze Pasted Content")
        analyze_fetch_button = False # Ensure fetch button state doesn't interfere

    else: # Fetch from URL
        # Text input for URL
        url = st.text_input("Enter the URL:", value=st.session_state.get('url', ''),
                           placeholder="e.g., https://www.example.com/your-page")
        # Update session state URL
        st.session_state.url = url
        # Clear pasted content state initially for URL mode on radio button change
        # This is handled implicitly by the input_method check below

        # Button to trigger fetch and then analysis
        analyze_fetch_button = st.button("Fetch and Analyze URL")
        analyze_pasted_button = False # Ensure paste button state doesn't interfere

        # Logic to fetch content when the fetch button is clicked
        if analyze_fetch_button and url:
             with st.spinner(f"Fetching content from {url}..."):
                fetched_content = fetch_content_from_url(url)
                if fetched_content:
                    st.session_state.content = fetched_content # Store fetched content
                    st.success("Content fetched successfully! Running analysis...")
                    st.session_state.fetch_button_clicked = True # Indicate successful fetch for the analysis trigger
                else:
                     st.error("Failed to fetch content. Please check the URL or try pasting.")
                     st.session_state.content = "" # Clear content on failure
                     st.session_state.url = "" # Clear URL on failure
                     st.session_state.fetch_button_clicked = False # Reset state on failure if fetch fails
             # Trigger a rerun after fetch attempt to update the UI and potentially start analysis
             raise RerunException(RerunData()) # Use the correct rerun method with RerunData

        elif analyze_fetch_button and not url:
             st.warning("Please enter a URL.")
             st.session_state.fetch_button_clicked = False # Reset state if button clicked with no URL

    # Display fetched content in a text area for review if available and in URL mode
    # This text area also allows editing the fetched content before analysis if needed
    if st.session_state.content and input_method == "Fetch from URL":
         st.subheader("Fetched Content (Review)")
         # Use a unique key for the text area in URL mode to maintain its state separately
         st.session_state.content = st.text_area("Review and edit fetched content:", value=st.session_state.content, height=300, key="fetched_content_review_area")

    # Enhanced content categorization section with keyword targeting
    st.write("### Content Categorization & SEO Targeting")
    st.write("Categorize your content and specify target keywords for more precise SEO analysis:")

    # Create three columns for the dropdown menus and keyword input
    cat_col1, cat_col2, cat_col3 = st.columns([1, 1, 1.2])

    with cat_col1:
        business_type = st.selectbox(
            "Business Type:",
            options=["Lead Generation/Service", "E-commerce", "SaaS/Tech", "Educational/Informational", "Local Business"],
            index=["Lead Generation/Service", "E-commerce", "SaaS/Tech", "Educational/Informational", "Local Business"].index(
                "Lead Generation/Service" if st.session_state.business_type == "lead_generation" else
                "E-commerce" if st.session_state.business_type == "ecommerce" else
                "SaaS/Tech" if st.session_state.business_type == "saas" else
                "Educational/Informational" if st.session_state.business_type == "educational" else
                "Local Business"
            ),
            help="Select the business model that best matches your content",
            key="business_type_selectbox"
        )

        # Convert display values to internal values
        st.session_state.business_type = (
            "lead_generation" if business_type == "Lead Generation/Service" else
            "ecommerce" if business_type == "E-commerce" else
            "saas" if business_type == "SaaS/Tech" else
            "educational" if business_type == "Educational/Informational" else
            "local_business"
        )

    # Dynamically change page type options based on business type
    with cat_col2:
        if st.session_state.business_type == "lead_generation":
            page_options = ["Landing Page", "Service Page", "Blog Post", "About Us", "Contact Page"]
        elif st.session_state.business_type == "ecommerce":
            page_options = ["Product Page", "Category Page", "Homepage", "Blog Post", "Checkout Page"]
        elif st.session_state.business_type == "saas":
            page_options = ["Feature Page", "Pricing Page", "Homepage", "Blog Post", "Documentation"]
        elif st.session_state.business_type == "educational":
            page_options = ["Course Page", "Resource Page", "Blog Post", "Homepage", "About Us"]
        else:  # local_business
            page_options = ["Homepage", "Service Page", "Location Page", "About Us", "Contact Page"]

        # Get current page type and find its index in the new options, defaulting to 0 if not found
        current_page_display = next((p for p in page_options if p.lower().replace(" ", "_") == st.session_state.page_type), page_options[0])
        current_page_index = page_options.index(current_page_display) if current_page_display in page_options else 0

        page_type = st.selectbox(
            "Page Type:",
            options=page_options,
            index=current_page_index,
            help="Select the type of page you're analyzing",
            key="page_type_selectbox"
        )

        # Convert display values to internal values
        st.session_state.page_type = page_type.lower().replace(" ", "_")

    # New keyword targeting input
    with cat_col3:
        target_keywords = st.text_area(
            "Target Keywords:",
            value=st.session_state.target_keywords,
            height=100,
            placeholder="Enter your target keywords, one per line:\n\ndigital marketing services\nSEO optimization\nbrand strategy\nlocal SEO",
            help="Enter the keywords you want to rank for. Put each keyword on a new line. Include both primary and secondary keywords.",
            key="target_keywords_input"
        )
        
        st.session_state.target_keywords = target_keywords
        
        # Show keyword count and validation
        if target_keywords.strip():
            keywords_list = [kw.strip() for kw in target_keywords.split('\n') if kw.strip()]
            st.caption(f"Keywords entered: {len(keywords_list)}")
            
            # Show a preview of keywords in an expander
            with st.expander("Preview Keywords", expanded=False):
                for i, kw in enumerate(keywords_list[:10], 1):  # Show first 10
                    st.write(f"{i}. {kw}")
                if len(keywords_list) > 10:
                    st.write(f"... and {len(keywords_list) - 10} more")
        else:
            st.caption("Add keywords for targeted analysis")

    # Add keyword strategy tips
    with st.expander("Keyword Strategy Tips", expanded=False):
        st.markdown("""
        **Effective Keyword Targeting:**
        - **Primary Keywords**: 1-3 main keywords you want to rank for
        - **Secondary Keywords**: 5-10 related terms and long-tail variations
        - **Local Keywords**: Include location-based terms for local businesses
        - **Intent-based Keywords**: Match keywords to user search intent
        
        **Examples by Business Type:**
        - **Lead Gen**: "digital marketing agency", "PPC management services"
        - **E-commerce**: "wireless headphones", "bluetooth earbuds under $100"
        - **SaaS**: "project management software", "team collaboration tools"
        - **Educational**: "online courses", "digital marketing certification"
        - **Local**: "dentist near me", "plumber in [city]"
        """)

    # --- Analysis Trigger Logic ---
    # Determine if analysis should be triggered
    # Trigger analysis if either the paste button is clicked and content is available,
    # OR if the fetch button was clicked in the previous run and content is now successfully loaded in session state.
    trigger_analysis = False

    # Case 1: Paste Content and Analyze button clicked
    if input_method == "Paste Content" and analyze_pasted_button and st.session_state.content:
         trigger_analysis = True

    # Case 2: Fetch from URL button clicked and fetch was successful (state updated in previous rerun)
    # We check fetch_button_clicked which was set True in the previous rerun upon successful fetch
    elif input_method == "Fetch from URL" and st.session_state.fetch_button_clicked and st.session_state.content:
         trigger_analysis = True
         # Reset the fetch_button_clicked state immediately after triggering analysis
         st.session_state.fetch_button_clicked = False

    # --- Analysis Execution ---
    # This block runs if the trigger_analysis flag is True based on the conditions above
    if trigger_analysis:

        # Reset PDF data if a new analysis is starting
        st.session_state.pdf_data = None
        st.session_state.pdf_generated = False
        st.session_state.claude_analysis = None # Clear previous analysis results

        # Check if API keys are provided before proceeding with analysis
        if not st.session_state.google_api_key or not st.session_state.anthropic_api_key:
            st.error("Please provide both Google API and Anthropic API keys in the sidebar.")
            # Reset analysis state if keys are missing to prevent displaying old results
            st.session_state.embedding = None
            st.session_state.analysis = None
            st.session_state.claude_analysis = None
            # No rerun needed here, the error message is sufficient to stop
            return

        with st.spinner("Analyzing content with keyword targeting... This may take a minute."):

            # Get embedding with progress indicator
            progress_text = st.empty()
            progress_text.text("Generating embedding from Google Gemini API...")
            st.session_state.embedding = get_embedding(st.session_state.content)

            # Generate analysis
            progress_text.text("Analyzing embedding patterns...")
            st.session_state.analysis = analyze_embedding(st.session_state.embedding)

            # Get Claude analysis with business, page type, and keyword context
            progress_text.text("Getting comprehensive SEO and keyword analysis from Claude...")
            # Increase content sent to Claude to include keyword analysis section
            claude_content_snippet = st.session_state.content[:7000]
            st.session_state.claude_analysis = analyze_with_claude(
                st.session_state.embedding,
                claude_content_snippet,
                st.session_state.business_type,
                st.session_state.page_type
            )
            progress_text.empty()

            # Display results only after successful analysis
            if st.session_state.claude_analysis and "Error getting analysis" not in st.session_state.claude_analysis:
                 st.success("Analysis complete with keyword targeting insights!")
                 # Trigger a rerun to display the results tabs
                 raise RerunException(RerunData())
            else:
                 st.error("Analysis failed. Please check your API keys and try again.")

    # Only display the results if we have a completed analysis in the session state
    # Check for both embedding and claude_analysis to ensure analysis is complete and not an error message
    if st.session_state.embedding is not None and st.session_state.claude_analysis is not None and "Error getting analysis" not in st.session_state.claude_analysis:
        # Create tabs for different sections with keyword insights
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Visualizations", "Metrics", "Keywords", "Clusters", "Analysis Report"])

        # Tab 1: Visualizations
        with tab1:
            st.subheader("Embedding Overview")
            fig1 = plot_embedding_overview(st.session_state.embedding)
            st.pyplot(fig1)
            plt.close(fig1)  # Release memory

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top Dimensions")
                fig2 = plot_top_dimensions(st.session_state.embedding)
                st.pyplot(fig2)
                plt.close(fig2)  # Release memory

            with col2:
                st.subheader("Activation Distribution")
                fig3 = plot_activation_histogram(st.session_state.embedding)
                st.pyplot(fig3)
                plt.close(fig3)  # Release memory

            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Dimension Clusters")
                fig4 = plot_dimension_clusters(st.session_state.embedding)
                st.pyplot(fig4)
                plt.close(fig4)  # Release memory

            with col4:
                st.subheader("PCA Visualization")
                fig5 = plot_pca(st.session_state.embedding)
                st.pyplot(fig5)
                plt.close(fig5)  # Release memory

        # Tab 2: Metrics
        with tab2:
            st.subheader("Key Metrics")
            metrics = st.session_state.analysis["metrics"]

            # Create a 3-column layout for metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Dimensions", metrics["dimension_count"])
                st.metric("Mean Value", f"{metrics['mean_value']:.6f}")
                st.metric("Standard Deviation", f"{metrics['std_dev']:.6f}")

            with col2:
                st.metric("Min Value", f"{metrics['min_value']:.6f} (dim {metrics['min_dimension']})")
                st.metric("Max Value", f"{metrics['max_value']:.6f} (dim {metrics['max_dimension']})")
                st.metric("Median Value", f"{metrics['median_value']:.6f}")

            with col3:
                st.metric("Positive Values", f"{metrics['positive_count']} ({metrics['positive_count']/metrics['dimension_count']*100:.2f}%)")
                st.metric("Negative Values", f"{metrics['negative_count']} ({metrics['negative_count']/metrics['dimension_count']*100:.2f}%)")
                st.metric("Significant Dimensions", f"{metrics['significant_dims']} (>0.1)")

        # Tab 3: NEW Keyword Analysis Tab
        with tab3:
            st.subheader("Keyword Targeting Analysis")
            
            if st.session_state.target_keywords.strip():
                keywords_list = [kw.strip() for kw in st.session_state.target_keywords.split('\n') if kw.strip()]
                
                # Keyword overview
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Keywords", len(keywords_list))
                with col2:
                    st.metric("Primary Keywords", min(3, len(keywords_list)))
                with col3:
                    st.metric("Long-tail Keywords", max(0, len(keywords_list) - 3))
                
                # Display keywords with analysis if content contains keyword analysis
                if "=== KEYWORD TARGETING ANALYSIS ===" in st.session_state.content:
                    st.subheader("Keyword Performance Summary")
                    
                    # Parse keyword analysis from content (this is a simplified approach)
                    # In a production app, you'd want to store this data separately
                    content_lines = st.session_state.content.split('\n')
                    keyword_section_start = False
                    current_keyword = None
                    
                    for line in content_lines:
                        if "=== KEYWORD TARGETING ANALYSIS ===" in line:
                            keyword_section_start = True
                            continue
                        elif "=== CONTENT STATISTICS ===" in line:
                            break
                        elif keyword_section_start and line.startswith("KEYWORD:"):
                            current_keyword = line.replace("KEYWORD:", "").strip()
                            with st.expander(f"Analysis for: {current_keyword}", expanded=False):
                                # Extract keyword data from the following lines
                                # This is a simplified version - you might want to enhance this
                                st.write("**Keyword Analysis:**")
                                st.write("• Detailed analysis available in the full report")
                                st.write("• Check the Analysis Report tab for complete insights")
                else:
                    st.info("Keyword analysis will be included in the next content fetch or analysis.")
                
                # Show keyword list
                st.subheader("Target Keywords List")
                for i, keyword in enumerate(keywords_list, 1):
                    keyword_type = "Primary" if i <= 3 else "Secondary"
                    st.write(f"{i}. **{keyword}** _{keyword_type}_")
                
            else:
                st.info("**No target keywords specified**")
                st.write("Add target keywords in the Content Categorization section to see detailed keyword analysis including:")
                st.write("• Keyword density analysis")
                st.write("• SEO element integration")
                st.write("• Semantic alignment scores") 
                st.write("• Optimization recommendations")
                
                with st.expander("Keyword Research Tips", expanded=False):
                    st.markdown("""
                    **How to choose effective target keywords:**
                    
                    1. **Primary Keywords (1-3)**: Main terms you want to rank for
                    2. **Secondary Keywords (5-10)**: Supporting and long-tail variations
                    3. **Intent Matching**: Choose keywords that match your page purpose
                    4. **Competition Analysis**: Balance search volume with ranking difficulty
                    
                    **Keyword Research Tools:**
                    - Google Keyword Planner
                    - SEMrush / Ahrefs
                    - Google Search Console
                    - Answer The Public
                    """)

        # Tab 4: Clusters
        with tab4:
            st.subheader("Dimension Clusters")

            if not st.session_state.analysis["clusters"]:
                st.info("No significant dimension clusters detected.")
            else:
                for cluster in st.session_state.analysis["clusters"]:
                    with st.expander(f"Cluster #{cluster['id']}: Dimensions {cluster['start_dim']}-{cluster['end_dim']}"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Size", f"{cluster['size']} dimensions")
                        col2.metric("Avg Value", f"{cluster['avg_value']:.6f}")
                        col3.metric("Max Value", f"{cluster['max_value']:.6f} (dim {cluster['max_dim']})")

        # Tab 5: Enhanced Claude Analysis with keyword context
        with tab5:
            # Show content categorization info with keywords
            business_display = {
                "lead_generation": "Lead Generation/Service",
                "ecommerce": "E-commerce",
                "saas": "SaaS/Tech",
                "educational": "Educational/Informational",
                "local_business": "Local Business"
            }.get(st.session_state.business_type, st.session_state.business_type.replace("_", " ").title())

            page_display = st.session_state.page_type.replace("_", " ").title()
            
            # Enhanced context display with keywords
            keywords_display = ""
            if st.session_state.target_keywords.strip():
                keywords_list = [kw.strip() for kw in st.session_state.target_keywords.split('\n') if kw.strip()]
                keywords_display = f"<p style=\"margin: 0;\"><strong>Target Keywords:</strong> {', '.join(keywords_list[:5])}"
                if len(keywords_list) > 5:
                    keywords_display += f" (+{len(keywords_list) - 5} more)"
                keywords_display += "</p>"
            else:
                keywords_display = "<p style=\"margin: 0;\"><strong>Target Keywords:</strong> <em>None specified</em></p>"

            st.markdown(f"""
            <div style="background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <p style="margin: 0; font-weight: bold; color: #1f4e79;">Analysis Context:</p>
                <p style="margin: 0;"><strong>Business Type:</strong> {business_display}</p>
                <p style="margin: 0;"><strong>Page Type:</strong> {page_display}</p>
                {keywords_display}
            </div>
            """, unsafe_allow_html=True)

            st.subheader("Comprehensive SEO & Keyword Analysis Report")

            # Display the analysis content
            st.markdown(st.session_state.claude_analysis, unsafe_allow_html=True)

            # Enhanced download options
            st.write("---")

            with st.container():
                st.markdown("""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
                <h3 style="margin-top: 0;">Download Options</h3>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                # Text download button
                with col1:
                    st.download_button(
                        label="Download as Text",
                        data=st.session_state.claude_analysis,
                        file_name="seo_keyword_analysis_report.txt",
                        mime="text/plain",
                        help="Download the analysis as a plain text file",
                        key="download_text_button"
                    )

                # PDF generation button and download
                with col2:
                    if st.session_state.pdf_data is None:
                        if st.button("Generate PDF Report", help="Create a PDF with visualizations and analysis", key="generate_pdf_button_tab"):
                            with st.spinner("Generating comprehensive PDF report with keyword analysis... This may take up to a minute."):
                                try:
                                    pdf_bytes = create_report_pdf(
                                        st.session_state.embedding,
                                        st.session_state.analysis,
                                        st.session_state.claude_analysis,
                                        st.session_state.business_type,
                                        st.session_state.page_type
                                    )
                                    st.session_state.pdf_data = pdf_bytes
                                    st.session_state.pdf_generated = True
                                    st.success("PDF report with keyword insights generated successfully!")
                                    # Trigger a rerun to show the download button immediately
                                    raise RerunException(RerunData())
                                except Exception as e:
                                    st.error(f"Error generating PDF: {str(e)}")
                    else:
                         # Show PDF download button if PDF has been generated
                         b64_pdf = base64.b64encode(st.session_state.pdf_data).decode('utf-8')
                         download_link = f'<a href="data:application/pdf;base64,{b64_pdf}" download="seo_keyword_analysis_report.pdf" class="button" style="display: inline-block; padding: 12px 20px; background-color: #0c6b58; color: white; text-decoration: none; font-weight: bold; border-radius: 4px; text-align: center; margin: 10px 0; width: 100%;">DOWNLOAD COMPLETE PDF REPORT</a>'
                         st.markdown(download_link, unsafe_allow_html=True)

                         if st.button("Generate New PDF", key="discard_pdf_button_tab"):
                            st.session_state.pdf_data = None
                            st.session_state.pdf_generated = False
                            # Trigger a rerun to update the button state
                            raise RerunException(RerunData())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        # Add a basic content area to ensure something is visible
        st.write("## SEO Embedding Analysis Tool")
        st.write("This tool analyzes content semantics using embeddings."), r'<strong>\1</strong>', analysis_text, flags=re.MULTILINE)
    return formatted_text

def plot_embedding_overview(embedding):
    """Create overview plot of embedding values"""
    try:
        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        ax.plot(range(len(embedding)), embedding)
        ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax.set_title('Embedding Values Across All 3k Dimensions')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        return fig
    except Exception as e:
        st.error(f"Error plotting embedding overview: {e}")
        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center')
        return fig

def plot_top_dimensions(embedding):
    """Plot top dimensions by magnitude"""
    try:
        # Get indices of top 20 dimensions by magnitude
        top_indices = sorted(range(len(embedding)), key=lambda i: abs(embedding[i]), reverse=True)[:20]
        top_values = [embedding[i] for i in top_indices]

        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        colors = ['blue' if v >= 0 else 'red' for v in top_values]
        ax.bar(range(len(top_indices)), top_values, color=colors)
        ax.set_xticks(range(len(top_indices)))
        ax.set_xticklabels(top_indices, rotation=45)
        ax.set_title('Top 20 Dimensions by Magnitude')
        ax.set_xlabel('Dimension Index')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        return fig
    except Exception as e:
        st.error(f"Error plotting top dimensions: {e}")
        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center')
        return fig

def plot_dimension_clusters(embedding):
    """Plot dimension clusters heatmap"""
    try:
        # Reshape embedding to highlight patterns
        embedding_reshaped = np.array(embedding).reshape(64, 48)

        fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
        # Create a custom colormap from blue to white to red
        cmap = LinearSegmentedColormap.from_list('BrBG', ['blue', 'white', 'red'], N=256)
        im = ax.imshow(embedding_reshaped, cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax, label='Activation Value')
        ax.set_title('Embedding Clusters Heatmap (Reshaped to 64x48)')
        ax.set_xlabel('Dimension Group')
        ax.set_ylabel('Dimension Group')
        return fig
    except Exception as e:
        st.error(f"Error plotting dimension clusters: {e}")
        fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center')
        return fig

def plot_pca(embedding):
    """Plot PCA visualization of embedding dimensions"""
    try:
        # Create a 2D array where each row is a segment of the original embedding
        segment_size = 256
        num_segments = len(embedding) // segment_size
        data_matrix = np.zeros((num_segments, segment_size))

        # Fill the matrix with segments
        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size
            data_matrix[i] = embedding[start:end]

        # Apply PCA
        fig = plt.figure(figsize=(10, 8))
        if num_segments > 1:
            pca = PCA(n_components=2)
            pca_results = pca.fit_transform(data_matrix)

            plt.scatter(pca_results[:, 0], pca_results[:, 1])

            # Label each point with its segment range
            for i in range(num_segments):
                start = i * segment_size
                end = start + segment_size - 1
                plt.annotate(f"{start}-{end}",
                            (pca_results[i, 0], pca_results[i, 1]),
                            fontsize=8)

            plt.title('PCA of Embedding Segments')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.grid(True, alpha=0.3)
        else:
            # If we don't have enough segments, create a simpler visualization
            plt.text(0.5, 0.5, "Not enough segments for PCA visualization",
                    ha='center', va='center', fontsize=12)
            plt.axis('off')

        return fig
    except Exception as e:
        st.error(f"Error plotting PCA: {e}")
        fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center')
        return fig

def plot_activation_histogram(embedding):
    """Plot histogram of embedding activation values"""
    try:
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        ax.hist(embedding, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        ax.set_title('Distribution of Embedding Values')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        return fig
    except Exception as e:
        st.error(f"Error plotting activation histogram: {e}")
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center')
        return fig

def analyze_embedding(embedding):
    """Analyze embedding for key metrics"""
    try:
        embedding = np.array(embedding)  # Convert to numpy array for easier processing
        abs_embedding = np.abs(embedding)

        # Calculate key metrics - CONVERT NUMPY TYPES TO PYTHON NATIVE TYPES
        metrics = {
            "dimension_count": int(len(embedding)),
            "mean_value": float(np.mean(embedding)),
            "std_dev": float(np.std(embedding)),
            "min_value": float(np.min(embedding)),
            "min_dimension": int(np.argmin(embedding)),
            "max_value": float(np.max(embedding)),
            "max_dimension": int(np.argmax(embedding)),
            "median_value": float(np.median(embedding)),
            "positive_count": int(np.sum(embedding > 0)),
            "negative_count": int(np.sum(embedding < 0)),
            "zero_count": int(np.sum(embedding == 0)),
            "abs_mean": float(np.mean(abs_embedding)),
            "significant_dims": int(np.sum(abs_embedding > 0.1))
        }

        # Find activation clusters
        significant_threshold = 0.1
        significant_dims = np.where(abs_embedding > significant_threshold)[0]

        # Find clusters (dimensions that are close to each other)
        clusters = []
        if len(significant_dims) > 0:
            current_cluster = [int(significant_dims[0])]  # Convert to int

            for i in range(1, len(significant_dims)):
                if significant_dims[i] - significant_dims[i-1] <= 5:  # If dimensions are close
                    current_cluster.append(int(significant_dims[i]))  # Convert to int
                else:
                    if len(current_cluster) > 0:
                        clusters.append(current_cluster)
                    current_cluster = [int(significant_dims[i])]  # Convert to int

            if len(current_cluster) > 0:
                clusters.append(current_cluster)

        # Filter to meaningful clusters (more than 1 dimension)
        clusters = [c for c in clusters if len(c) > 1]

        # Format clusters for display
        cluster_info = []
        for i, cluster in enumerate(clusters):
            values = [float(embedding[dim]) for dim in cluster]  # Convert to float
            cluster_info.append({
                "id": i+1,
                "dimensions": [int(d) for d in cluster],  # Convert to int
                "start_dim": int(min(cluster)),
                "end_dim": int(max(cluster)),
                "size": int(len(cluster)),
                "avg_value": float(np.mean(values)),
                "max_value": float(np.max(values)),
                "max_dim": int(cluster[np.argmax(values)])
            })

        # Top dimensions by magnitude
        top_indices = sorted(range(len(embedding)), key=lambda i: abs(embedding[i]), reverse=True)[:10]
        top_dimensions = [{"dimension": int(idx), "value": float(embedding[idx])} for idx in top_indices]

        return {
            "metrics": metrics,
            "clusters": cluster_info,
            "top_dimensions": top_dimensions
        }
    except Exception as e:
        st.error(f"Error analyzing embedding: {e}")
        # Return a minimal valid structure in case of error
        return {
            "metrics": {
                "dimension_count": 0,
                "mean_value": 0.0,
                "std_dev": 0.0,
                "min_value": 0.0,
                "min_dimension": 0,
                "max_value": 0.0,
                "max_dimension": 0,
                "median_value": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "zero_count": 0,
                "abs_mean": 0.0,
                "significant_dims": 0
            },
            "clusters": [],
            "top_dimensions": []
        }

def create_report_pdf(embedding, analysis, claude_analysis, business_type, page_type):
    """Create a better-formatted PDF report of the analysis results with business and page type context"""
    try:
        # Create a buffer for the PDF
        buffer = BytesIO()

        # Create the PDF document with better margins
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )

        # Create custom styles
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading1_style = styles['Heading1']
        heading2_style = styles['Heading2']

        # Create a better normal style with more space
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            spaceBefore=6,
            spaceAfter=6,
            leading=14  # Line spacing
        )

        # Create a story (content)
        story = []

        # Add title with better spacing
        story.append(Paragraph("SEO Embedding Analysis Report", title_style))
        story.append(Spacer(1, 24))  # More space after title

        # Add date with better formatting
        from datetime import datetime
        date_style = ParagraphStyle(
            'DateStyle',
            parent=normal_style,
            fontName='Helvetica-Oblique',
            textColor=colors.darkblue
        )
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", date_style))
        story.append(Spacer(1, 12))  # Space after date

        # Add content categorization information
        category_style = ParagraphStyle(
            'CategoryStyle',
            parent=normal_style,
            fontName='Helvetica-Bold',
            textColor=colors.darkblue,
            fontSize=11
        )

        # Convert internal values to display values
        business_display = {
            "lead_generation": "Lead Generation/Service",
            "ecommerce": "E-commerce",
            "saas": "SaaS/Tech",
            "educational": "Educational/Informational",
            "local_business": "Local Business"
        }.get(business_type, business_type.replace("_", " ").title())

        page_display = page_type.replace("_", " ").title()

        story.append(Paragraph(f"Business Type: {business_display}", category_style))
        story.append(Paragraph(f"Page Type: {page_display}", category_style))
        
        # Add keyword information if available
        if st.session_state.target_keywords.strip():
            keywords_list = [kw.strip() for kw in st.session_state.target_keywords.split('\n') if kw.strip()]
            keywords_text = ", ".join(keywords_list[:5])
            if len(keywords_list) > 5:
                keywords_text += f" (+{len(keywords_list) - 5} more)"
            story.append(Paragraph(f"Target Keywords: {keywords_text}", category_style))
        
        story.append(Spacer(1, 24))  # More space after categories

        # Add metrics section with better formatting
        story.append(Paragraph("Key Metrics", heading1_style))
        story.append(Spacer(1, 12))

        # Format metrics as a table for better appearance
        metrics = analysis["metrics"]
        metrics_data = [
            ["Metric", "Value"],
            ["Dimensions", f"{metrics['dimension_count']}"],
            ["Mean Value", f"{metrics['mean_value']:.6f}"],
            ["Standard Deviation", f"{metrics['std_dev']:.6f}"],
            ["Min Value", f"{metrics['min_value']:.6f} (dim {metrics['min_dimension']})"],
            ["Max Value", f"{metrics['max_value']:.6f} (dim {metrics['max_dimension']})"],
            ["Positive Values", f"{metrics['positive_count']} ({metrics['positive_count']/metrics['dimension_count']*100:.2f}%)"],
            ["Negative Values", f"{metrics['negative_count']} ({metrics['negative_count']/metrics['dimension_count']*100:.2f}%)"],
            ["Significant Dimensions", f"{metrics['significant_dims']} (>0.1)"]
        ]

        # Create a table with the metrics
        metrics_table = Table(metrics_data, colWidths=[2*inch, 3.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 24))

        # Create visualizations page with better layout
        story.append(PageBreak())
        story.append(Paragraph("Embedding Visualizations", heading1_style))
        story.append(Spacer(1, 16))

        # Function to add an image with a better caption
        def add_figure_with_caption(story, img_data, caption):
            # Add caption with better styling
            caption_style = ParagraphStyle(
                'CaptionStyle',
                parent=heading2_style,
                textColor=colors.darkblue,
                spaceAfter=8
            )
            story.append(Paragraph(caption, caption_style))

            # Create image with better size settings
            img = Image(img_data, width=7*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 16))

        # Create plots with better resolution and save them as images
        # 1. Embedding Overview
        fig_overview = plot_embedding_overview(embedding)
        fig_overview.set_dpi(150)  # Higher resolution
        fig_overview.tight_layout()  # Better layout
        overview_img = BytesIO()
        fig_overview.savefig(overview_img, format='png', bbox_inches='tight')
        plt.close(fig_overview)
        overview_img.seek(0)

        # 2. Top Dimensions
        fig_top = plot_top_dimensions(embedding)
        fig_top.set_dpi(150)
        fig_top.tight_layout()
        top_img = BytesIO()
        fig_top.savefig(top_img, format='png', bbox_inches='tight')
        plt.close(fig_top)
        top_img.seek(0)

        # 3. Activation Histogram
        fig_hist = plot_activation_histogram(embedding)
        fig_hist.set_dpi(150)
        fig_hist.tight_layout()
        hist_img = BytesIO()
        fig_hist.savefig(hist_img, format='png', bbox_inches='tight')
        plt.close(fig_hist)
        hist_img.seek(0)

        # Add the first three visualizations
        add_figure_with_caption(story, overview_img, "Embedding Overview")
        add_figure_with_caption(story, top_img, "Top 20 Dimensions by Magnitude")
        add_figure_with_caption(story, hist_img, "Distribution of Embedding Values")

        # Start a new page for the next visualizations
        story.append(PageBreak())

        # 4. Dimension Clusters
        fig_clusters = plot_dimension_clusters(embedding)
        fig_clusters.set_dpi(150)
        fig_clusters.tight_layout()
        clusters_img = BytesIO()
        fig_clusters.savefig(clusters_img, format='png', bbox_inches='tight')
        plt.close(fig_clusters)
        clusters_img.seek(0)

        # 5. PCA Visualization
        fig_pca = plot_pca(embedding)
        fig_pca.set_dpi(150)
        fig_pca.tight_layout()
        pca_img = BytesIO()
        fig_pca.savefig(pca_img, format='png', bbox_inches='tight')
        plt.close(fig_pca)
        pca_img.seek(0)

        # Add the remaining visualizations
        add_figure_with_caption(story, clusters_img, "Dimension Clusters Heatmap")
        add_figure_with_caption(story, pca_img, "PCA Visualization of Embedding Segments")

        # Add Claude Analysis section with better formatting
        story.append(PageBreak())
        story.append(Paragraph("Comprehensive Analysis", heading1_style))
        story.append(Spacer(1, 16))

        # Add categorization banner for context
        context_style = ParagraphStyle(
            'ContextStyle',
            parent=normal_style,
            fontName='Helvetica-Bold',
            textColor=colors.white,
            fontSize=10,
            alignment=1,  # Center alignment
            backColor=colors.darkblue
        )

        context_text = f"Analysis tailored for: {business_display} | {page_display}"
        if st.session_state.target_keywords.strip():
            keywords_list = [kw.strip() for kw in st.session_state.target_keywords.split('\n') if kw.strip()]
            context_text += f" | Keywords: {len(keywords_list)} targeted"

        story.append(Paragraph(context_text, context_style))
        story.append(Spacer(1, 12))

        # Process Claude analysis text - convert from HTML/markdown to plain text
        # Remove HTML tags
        analysis_text = re.sub(r'<[^>]*>', '', claude_analysis)

        # Function to process headings better
        def process_heading(text, style):
            text = text.strip()
            if text.startswith('##'):
                return Paragraph(text.replace('##', '').strip(), heading1_style)
            elif text.startswith('#'):
                return Paragraph(text.replace('#', '').strip(), heading2_style)
            elif text.strip() and text.strip()[0].isupper():  # Likely a section title without # marks
                first_line = text.split("\n")[0].strip()
                if len(first_line) < 50 and first_line.isupper():  # It's probably a heading
                    return Paragraph(first_line, heading2_style)
            # Normal paragraph
            return Paragraph(text, normal_style)

        # Split by paragraphs and process each one
        paragraphs = analysis_text.split('\n\n')

        for paragraph in paragraphs:
            if paragraph.strip():
                element = process_heading(paragraph, normal_style)
                story.append(element)
                if isinstance(element, Paragraph) and element.style == heading1_style:
                    story.append(Spacer(1, 12))  # More space after main headings

        # Build the PDF
        doc.build(story)

        # Get the PDF value
        pdf_value = buffer.getvalue()
        buffer.close()

        return pdf_value

    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

# Main content area
def main():
    st.title("SEO Embedding Analysis Tool")

    # Content input options
    st.header("Content Input")

    input_method = st.radio("Choose input method:", ("Paste Content", "Fetch from URL"), key="input_method_radio")

    # Use a placeholder for content that reflects the current input method
    content_placeholder = "Enter the content you want to analyze..." if input_method == "Paste Content" else "Content will be fetched from the URL..."

    if input_method == "Paste Content":
        # Text area for pasting content
        # Value is read directly from session state on rerun
        st.session_state.content = st.text_area("Paste your content here:", height=300,
                              placeholder=content_placeholder,
                              value=st.session_state.get('content', ''), key="pasted_content_area")
        st.session_state.url = "" # Clear URL state when pasting content
        # Button for analyzing pasted content
        analyze_pasted_button = st.button("Analyze Pasted Content")
        analyze_fetch_button = False # Ensure fetch button state doesn't interfere

    else: # Fetch from URL
        # Text input for URL
        url = st.text_input("Enter the URL:", value=st.session_state.get('url', ''),
                           placeholder="e.g., https://www.example.com/your-page")
        # Update session state URL
        st.session_state.url = url
        # Clear pasted content state initially for URL mode on radio button change
        # This is handled implicitly by the input_method check below

        # Button to trigger fetch and then analysis
        analyze_fetch_button = st.button("Fetch and Analyze URL")
        analyze_pasted_button = False # Ensure paste button state doesn't interfere

        # Logic to fetch content when the fetch button is clicked
        if analyze_fetch_button and url:
             with st.spinner(f"Fetching content from {url}..."):
                fetched_content = fetch_content_from_url(url)
                if fetched_content:
                    st.session_state.content = fetched_content # Store fetched content
                    st.success("Content fetched successfully! Running analysis...")
                    st.session_state.fetch_button_clicked = True # Indicate successful fetch for the analysis trigger
                else:
                     st.error("Failed to fetch content. Please check the URL or try pasting.")
                     st.session_state.content = "" # Clear content on failure
                     st.session_state.url = "" # Clear URL on failure
                     st.session_state.fetch_button_clicked = False # Reset state on failure if fetch fails
             # Trigger a rerun after fetch attempt to update the UI and potentially start analysis
             raise RerunException(RerunData()) # Use the correct rerun method with RerunData

        elif analyze_fetch_button and not url:
             st.warning("Please enter a URL.")
             st.session_state.fetch_button_clicked = False # Reset state if button clicked with no URL

    # Display fetched content in a text area for review if available and in URL mode
    # This text area also allows editing the fetched content before analysis if needed
    if st.session_state.content and input_method == "Fetch from URL":
         st.subheader("Fetched Content (Review)")
         # Use a unique key for the text area in URL mode to maintain its state separately
         st.session_state.content = st.text_area("Review and edit fetched content:", value=st.session_state.content, height=300, key="fetched_content_review_area")

    # Enhanced content categorization section with keyword targeting
    st.write("### Content Categorization & SEO Targeting")
    st.write("Categorize your content and specify target keywords for more precise SEO analysis:")

    # Create three columns for the dropdown menus and keyword input
    cat_col1, cat_col2, cat_col3 = st.columns([1, 1, 1.2])

    with cat_col1:
        business_type = st.selectbox(
            "Business Type:",
            options=["Lead Generation/Service", "E-commerce", "SaaS/Tech", "Educational/Informational", "Local Business"],
            index=["Lead Generation/Service", "E-commerce", "SaaS/Tech", "Educational/Informational", "Local Business"].index(
                "Lead Generation/Service" if st.session_state.business_type == "lead_generation" else
                "E-commerce" if st.session_state.business_type == "ecommerce" else
                "SaaS/Tech" if st.session_state.business_type == "saas" else
                "Educational/Informational" if st.session_state.business_type == "educational" else
                "Local Business"
            ),
            help="Select the business model that best matches your content",
            key="business_type_selectbox"
        )

        # Convert display values to internal values
        st.session_state.business_type = (
            "lead_generation" if business_type == "Lead Generation/Service" else
            "ecommerce" if business_type == "E-commerce" else
            "saas" if business_type == "SaaS/Tech" else
            "educational" if business_type == "Educational/Informational" else
            "local_business"
        )

    # Dynamically change page type options based on business type
    with cat_col2:
        if st.session_state.business_type == "lead_generation":
            page_options = ["Landing Page", "Service Page", "Blog Post", "About Us", "Contact Page"]
        elif st.session_state.business_type == "ecommerce":
            page_options = ["Product Page", "Category Page", "Homepage", "Blog Post", "Checkout Page"]
        elif st.session_state.business_type == "saas":
            page_options = ["Feature Page", "Pricing Page", "Homepage", "Blog Post", "Documentation"]
        elif st.session_state.business_type == "educational":
            page_options = ["Course Page", "Resource Page", "Blog Post", "Homepage", "About Us"]
        else:  # local_business
            page_options = ["Homepage", "Service Page", "Location Page", "About Us", "Contact Page"]

        # Get current page type and find its index in the new options, defaulting to 0 if not found
        current_page_display = next((p for p in page_options if p.lower().replace(" ", "_") == st.session_state.page_type), page_options[0])
        current_page_index = page_options.index(current_page_display) if current_page_display in page_options else 0

        page_type = st.selectbox(
            "Page Type:",
            options=page_options,
            index=current_page_index,
            help="Select the type of page you're analyzing",
            key="page_type_selectbox"
        )

        # Convert display values to internal values
        st.session_state.page_type = page_type.lower().replace(" ", "_")

    # New keyword targeting input
    with cat_col3:
        target_keywords = st.text_area(
            "Target Keywords:",
            value=st.session_state.target_keywords,
            height=100,
            placeholder="Enter your target keywords, one per line:\n\ndigital marketing services\nSEO optimization\nbrand strategy\nlocal SEO",
            help="Enter the keywords you want to rank for. Put each keyword on a new line. Include both primary and secondary keywords.",
            key="target_keywords_input"
        )
        
        st.session_state.target_keywords = target_keywords
        
        # Show keyword count and validation
        if target_keywords.strip():
            keywords_list = [kw.strip() for kw in target_keywords.split('\n') if kw.strip()]
            st.caption(f"Keywords entered: {len(keywords_list)}")
            
            # Show a preview of keywords in an expander
            with st.expander("Preview Keywords", expanded=False):
                for i, kw in enumerate(keywords_list[:10], 1):  # Show first 10
                    st.write(f"{i}. {kw}")
                if len(keywords_list) > 10:
                    st.write(f"... and {len(keywords_list) - 10} more")
        else:
            st.caption("Add keywords for targeted analysis")

    # Add keyword strategy tips
    with st.expander("Keyword Strategy Tips", expanded=False):
        st.markdown("""
        **Effective Keyword Targeting:**
        - **Primary Keywords**: 1-3 main keywords you want to rank for
        - **Secondary Keywords**: 5-10 related terms and long-tail variations
        - **Local Keywords**: Include location-based terms for local businesses
        - **Intent-based Keywords**: Match keywords to user search intent
        
        **Examples by Business Type:**
        - **Lead Gen**: "digital marketing agency", "PPC management services"
        - **E-commerce**: "wireless headphones", "bluetooth earbuds under $100"
        - **SaaS**: "project management software", "team collaboration tools"
        - **Educational**: "online courses", "digital marketing certification"
        - **Local**: "dentist near me", "plumber in [city]"
        """)

    # --- Analysis Trigger Logic ---
    # Determine if analysis should be triggered
    # Trigger analysis if either the paste button is clicked and content is available,
    # OR if the fetch button was clicked in the previous run and content is now successfully loaded in session state.
    trigger_analysis = False

    # Case 1: Paste Content and Analyze button clicked
    if input_method == "Paste Content" and analyze_pasted_button and st.session_state.content:
         trigger_analysis = True

    # Case 2: Fetch from URL button clicked and fetch was successful (state updated in previous rerun)
    # We check fetch_button_clicked which was set True in the previous rerun upon successful fetch
    elif input_method == "Fetch from URL" and st.session_state.fetch_button_clicked and st.session_state.content:
         trigger_analysis = True
         # Reset the fetch_button_clicked state immediately after triggering analysis
         st.session_state.fetch_button_clicked = False

    # --- Analysis Execution ---
    # This block runs if the trigger_analysis flag is True based on the conditions above
    if trigger_analysis:

        # Reset PDF data if a new analysis is starting
        st.session_state.pdf_data = None
        st.session_state.pdf_generated = False
        st.session_state.claude_analysis = None # Clear previous analysis results

        # Check if API keys are provided before proceeding with analysis
        if not st.session_state.google_api_key or not st.session_state.anthropic_api_key:
            st.error("Please provide both Google API and Anthropic API keys in the sidebar.")
            # Reset analysis state if keys are missing to prevent displaying old results
            st.session_state.embedding = None
            st.session_state.analysis = None
            st.session_state.claude_analysis = None
            # No rerun needed here, the error message is sufficient to stop
            return

        with st.spinner("Analyzing content with keyword targeting... This may take a minute."):

            # Get embedding with progress indicator
            progress_text = st.empty()
            progress_text.text("Generating embedding from Google Gemini API...")
            st.session_state.embedding = get_embedding(st.session_state.content)

            # Generate analysis
            progress_text.text("Analyzing embedding patterns...")
            st.session_state.analysis = analyze_embedding(st.session_state.embedding)

            # Get Claude analysis with business, page type, and keyword context
            progress_text.text("Getting comprehensive SEO and keyword analysis from Claude...")
            # Increase content sent to Claude to include keyword analysis section
            claude_content_snippet = st.session_state.content[:7000]
            st.session_state.claude_analysis = analyze_with_claude(
                st.session_state.embedding,
                claude_content_snippet,
                st.session_state.business_type,
                st.session_state.page_type
            )
            progress_text.empty()

            # Display results only after successful analysis
            if st.session_state.claude_analysis and "Error getting analysis" not in st.session_state.claude_analysis:
                 st.success("Analysis complete with keyword targeting insights!")
                 # Trigger a rerun to display the results tabs
                 raise RerunException(RerunData())
            else:
                 st.error("Analysis failed. Please check your API keys and try again.")

    # Only display the results if we have a completed analysis in the session state
    # Check for both embedding and claude_analysis to ensure analysis is complete and not an error message
    if st.session_state.embedding is not None and st.session_state.claude_analysis is not None and "Error getting analysis" not in st.session_state.claude_analysis:
        # Create tabs for different sections with keyword insights
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Visualizations", "Metrics", "Keywords", "Clusters", "Analysis Report"])

        # Tab 1: Visualizations
        with tab1:
            st.subheader("Embedding Overview")
            fig1 = plot_embedding_overview(st.session_state.embedding)
            st.pyplot(fig1)
            plt.close(fig1)  # Release memory

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top Dimensions")
                fig2 = plot_top_dimensions(st.session_state.embedding)
                st.pyplot(fig2)
                plt.close(fig2)  # Release memory

            with col2:
                st.subheader("Activation Distribution")
                fig3 = plot_activation_histogram(st.session_state.embedding)
                st.pyplot(fig3)
                plt.close(fig3)  # Release memory

            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Dimension Clusters")
                fig4 = plot_dimension_clusters(st.session_state.embedding)
                st.pyplot(fig4)
                plt.close(fig4)  # Release memory

            with col4:
                st.subheader("PCA Visualization")
                fig5 = plot_pca(st.session_state.embedding)
                st.pyplot(fig5)
                plt.close(fig5)  # Release memory

        # Tab 2: Metrics
        with tab2:
            st.subheader("Key Metrics")
            metrics = st.session_state.analysis["metrics"]

            # Create a 3-column layout for metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Dimensions", metrics["dimension_count"])
                st.metric("Mean Value", f"{metrics['mean_value']:.6f}")
                st.metric("Standard Deviation", f"{metrics['std_dev']:.6f}")

            with col2:
                st.metric("Min Value", f"{metrics['min_value']:.6f} (dim {metrics['min_dimension']})")
                st.metric("Max Value", f"{metrics['max_value']:.6f} (dim {metrics['max_dimension']})")
                st.metric("Median Value", f"{metrics['median_value']:.6f}")

            with col3:
                st.metric("Positive Values", f"{metrics['positive_count']} ({metrics['positive_count']/metrics['dimension_count']*100:.2f}%)")
                st.metric("Negative Values", f"{metrics['negative_count']} ({metrics['negative_count']/metrics['dimension_count']*100:.2f}%)")
                st.metric("Significant Dimensions", f"{metrics['significant_dims']} (>0.1)")

        # Tab 3: NEW Keyword Analysis Tab
        with tab3:
            st.subheader("Keyword Targeting Analysis")
            
            if st.session_state.target_keywords.strip():
                keywords_list = [kw.strip() for kw in st.session_state.target_keywords.split('\n') if kw.strip()]
                
                # Keyword overview
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Keywords", len(keywords_list))
                with col2:
                    st.metric("Primary Keywords", min(3, len(keywords_list)))
                with col3:
                    st.metric("Long-tail Keywords", max(0, len(keywords_list) - 3))
                
                # Display keywords with analysis if content contains keyword analysis
                if "=== KEYWORD TARGETING ANALYSIS ===" in st.session_state.content:
                    st.subheader("Keyword Performance Summary")
                    
                    # Parse keyword analysis from content (this is a simplified approach)
                    # In a production app, you'd want to store this data separately
                    content_lines = st.session_state.content.split('\n')
                    keyword_section_start = False
                    current_keyword = None
                    
                    for line in content_lines:
                        if "=== KEYWORD TARGETING ANALYSIS ===" in line:
                            keyword_section_start = True
                            continue
                        elif "=== CONTENT STATISTICS ===" in line:
                            break
                        elif keyword_section_start and line.startswith("KEYWORD:"):
                            current_keyword = line.replace("KEYWORD:", "").strip()
                            with st.expander(f"Analysis for: {current_keyword}", expanded=False):
                                # Extract keyword data from the following lines
                                # This is a simplified version - you might want to enhance this
                                st.write("**Keyword Analysis:**")
                                st.write("• Detailed analysis available in the full report")
                                st.write("• Check the Analysis Report tab for complete insights")
                else:
                    st.info("Keyword analysis will be included in the next content fetch or analysis.")
                
                # Show keyword list
                st.subheader("Target Keywords List")
                for i, keyword in enumerate(keywords_list, 1):
                    keyword_type = "Primary" if i <= 3 else "Secondary"
                    st.write(f"{i}. **{keyword}** _{keyword_type}_")
                
            else:
                st.info("**No target keywords specified**")
                st.write("Add target keywords in the Content Categorization section to see detailed keyword analysis including:")
                st.write("• Keyword density analysis")
                st.write("• SEO element integration")
                st.write("• Semantic alignment scores") 
                st.write("• Optimization recommendations")
                
                with st.expander("Keyword Research Tips", expanded=False):
                    st.markdown("""
                    **How to choose effective target keywords:**
                    
                    1. **Primary Keywords (1-3)**: Main terms you want to rank for
                    2. **Secondary Keywords (5-10)**: Supporting and long-tail variations
                    3. **Intent Matching**: Choose keywords that match your page purpose
                    4. **Competition Analysis**: Balance search volume with ranking difficulty
                    
                    **Keyword Research Tools:**
                    - Google Keyword Planner
                    - SEMrush / Ahrefs
                    - Google Search Console
                    - Answer The Public
                    """)

        # Tab 4: Clusters
        with tab4:
            st.subheader("Dimension Clusters")

            if not st.session_state.analysis["clusters"]:
                st.info("No significant dimension clusters detected.")
            else:
                for cluster in st.session_state.analysis["clusters"]:
                    with st.expander(f"Cluster #{cluster['id']}: Dimensions {cluster['start_dim']}-{cluster['end_dim']}"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Size", f"{cluster['size']} dimensions")
                        col2.metric("Avg Value", f"{cluster['avg_value']:.6f}")
                        col3.metric("Max Value", f"{cluster['max_value']:.6f} (dim {cluster['max_dim']})")

        # Tab 5: Enhanced Claude Analysis with keyword context
        with tab5:
            # Show content categorization info with keywords
            business_display = {
                "lead_generation": "Lead Generation/Service",
                "ecommerce": "E-commerce",
                "saas": "SaaS/Tech",
                "educational": "Educational/Informational",
                "local_business": "Local Business"
            }.get(st.session_state.business_type, st.session_state.business_type.replace("_", " ").title())

            page_display = st.session_state.page_type.replace("_", " ").title()
            
            # Enhanced context display with keywords
            keywords_display = ""
            if st.session_state.target_keywords.strip():
                keywords_list = [kw.strip() for kw in st.session_state.target_keywords.split('\n') if kw.strip()]
                keywords_display = f"<p style=\"margin: 0;\"><strong>Target Keywords:</strong> {', '.join(keywords_list[:5])}"
                if len(keywords_list) > 5:
                    keywords_display += f" (+{len(keywords_list) - 5} more)"
                keywords_display += "</p>"
            else:
                keywords_display = "<p style=\"margin: 0;\"><strong>Target Keywords:</strong> <em>None specified</em></p>"

            st.markdown(f"""
            <div style="background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <p style="margin: 0; font-weight: bold; color: #1f4e79;">Analysis Context:</p>
                <p style="margin: 0;"><strong>Business Type:</strong> {business_display}</p>
                <p style="margin: 0;"><strong>Page Type:</strong> {page_display}</p>
                {keywords_display}
            </div>
            """, unsafe_allow_html=True)

            st.subheader("Comprehensive SEO & Keyword Analysis Report")

            # Display the analysis content
            st.markdown(st.session_state.claude_analysis, unsafe_allow_html=True)

            # Enhanced download options
            st.write("---")

            with st.container():
                st.markdown("""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
                <h3 style="margin-top: 0;">Download Options</h3>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                # Text download button
                with col1:
                    st.download_button(
                        label="Download as Text",
                        data=st.session_state.claude_analysis,
                        file_name="seo_keyword_analysis_report.txt",
                        mime="text/plain",
                        help="Download the analysis as a plain text file",
                        key="download_text_button"
                    )

                # PDF generation button and download
                with col2:
                    if st.session_state.pdf_data is None:
                        if st.button("Generate PDF Report", help="Create a PDF with visualizations and analysis", key="generate_pdf_button_tab"):
                            with st.spinner("Generating comprehensive PDF report with keyword analysis... This may take up to a minute."):
                                try:
                                    pdf_bytes = create_report_pdf(
                                        st.session_state.embedding,
                                        st.session_state.analysis,
                                        st.session_state.claude_analysis,
                                        st.session_state.business_type,
                                        st.session_state.page_type
                                    )
                                    st.session_state.pdf_data = pdf_bytes
                                    st.session_state.pdf_generated = True
                                    st.success("PDF report with keyword insights generated successfully!")
                                    # Trigger a rerun to show the download button immediately
                                    raise RerunException(RerunData())
                                except Exception as e:
                                    st.error(f"Error generating PDF: {str(e)}")
                    else:
                         # Show PDF download button if PDF has been generated
                         b64_pdf = base64.b64encode(st.session_state.pdf_data).decode('utf-8')
                         download_link = f'<a href="data:application/pdf;base64,{b64_pdf}" download="seo_keyword_analysis_report.pdf" class="button" style="display: inline-block; padding: 12px 20px; background-color: #0c6b58; color: white; text-decoration: none; font-weight: bold; border-radius: 4px; text-align: center; margin: 10px 0; width: 100%;">DOWNLOAD COMPLETE PDF REPORT</a>'
                         st.markdown(download_link, unsafe_allow_html=True)

                         if st.button("Generate New PDF", key="discard_pdf_button_tab"):
                            st.session_state.pdf_data = None
                            st.session_state.pdf_generated = False
                            # Trigger a rerun to update the button state
                            raise RerunException(RerunData())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        # Add a basic content area to ensure something is visible
        st.write("## SEO Embedding Analysis Tool")
        st.write("This tool analyzes content semantics using embeddings."), r'<strong>\1</strong>', analysis_text, flags=re.MULTILINE)
    return formatted_text

def plot_embedding_overview(embedding):
    """Create overview plot of embedding values"""
    try:
        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        ax.plot(range(len(embedding)), embedding)
        ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax.set_title('Embedding Values Across All 3k Dimensions')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        return fig
    except Exception as e:
        st.error(f"Error plotting embedding overview: {e}")
        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center')
        return fig

def plot_top_dimensions(embedding):
    """Plot top dimensions by magnitude"""
    try:
        # Get indices of top 20 dimensions by magnitude
        top_indices = sorted(range(len(embedding)), key=lambda i: abs(embedding[i]), reverse=True)[:20]
        top_values = [embedding[i] for i in top_indices]

        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        colors = ['blue' if v >= 0 else 'red' for v in top_values]
        ax.bar(range(len(top_indices)), top_values, color=colors)
        ax.set_xticks(range(len(top_indices)))
        ax.set_xticklabels(top_indices, rotation=45)
        ax.set_title('Top 20 Dimensions by Magnitude')
        ax.set_xlabel('Dimension Index')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        return fig
    except Exception as e:
        st.error(f"Error plotting top dimensions: {e}")
        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center')
        return fig

def plot_dimension_clusters(embedding):
    """Plot dimension clusters heatmap"""
    try:
        # Reshape embedding to highlight patterns
        embedding_reshaped = np.array(embedding).reshape(64, 48)

        fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
        # Create a custom colormap from blue to white to red
        cmap = LinearSegmentedColormap.from_list('BrBG', ['blue', 'white', 'red'], N=256)
        im = ax.imshow(embedding_reshaped, cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax, label='Activation Value')
        ax.set_title('Embedding Clusters Heatmap (Reshaped to 64x48)')
        ax.set_xlabel('Dimension Group')
        ax.set_ylabel('Dimension Group')
        return fig
    except Exception as e:
        st.error(f"Error plotting dimension clusters: {e}")
        fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center')
        return fig

def plot_pca(embedding):
    """Plot PCA visualization of embedding dimensions"""
    try:
        # Create a 2D array where each row is a segment of the original embedding
        segment_size = 256
        num_segments = len(embedding) // segment_size
        data_matrix = np.zeros((num_segments, segment_size))

        # Fill the matrix with segments
        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size
            data_matrix[i] = embedding[start:end]

        # Apply PCA
        fig = plt.figure(figsize=(10, 8))
        if num_segments > 1:
            pca = PCA(n_components=2)
            pca_results = pca.fit_transform(data_matrix)

            plt.scatter(pca_results[:, 0], pca_results[:, 1])

            # Label each point with its segment range
            for i in range(num_segments):
                start = i * segment_size
                end = start + segment_size - 1
                plt.annotate(f"{start}-{end}",
                            (pca_results[i, 0], pca_results[i, 1]),
                            fontsize=8)

            plt.title('PCA of Embedding Segments')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.grid(True, alpha=0.3)
        else:
            # If we don't have enough segments, create a simpler visualization
            plt.text(0.5, 0.5, "Not enough segments for PCA visualization",
                    ha='center', va='center', fontsize=12)
            plt.axis('off')

        return fig
    except Exception as e:
        st.error(f"Error plotting PCA: {e}")
        fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center')
        return fig

def plot_activation_histogram(embedding):
    """Plot histogram of embedding activation values"""
    try:
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        ax.hist(embedding, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        ax.set_title('Distribution of Embedding Values')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        return fig
    except Exception as e:
        st.error(f"Error plotting activation histogram: {e}")
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center')
        return fig

def analyze_embedding(embedding):
    """Analyze embedding for key metrics"""
    try:
        embedding = np.array(embedding)  # Convert to numpy array for easier processing
        abs_embedding = np.abs(embedding)

        # Calculate key metrics - CONVERT NUMPY TYPES TO PYTHON NATIVE TYPES
        metrics = {
            "dimension_count": int(len(embedding)),
            "mean_value": float(np.mean(embedding)),
            "std_dev": float(np.std(embedding)),
            "min_value": float(np.min(embedding)),
            "min_dimension": int(np.argmin(embedding)),
            "max_value": float(np.max(embedding)),
            "max_dimension": int(np.argmax(embedding)),
            "median_value": float(np.median(embedding)),
            "positive_count": int(np.sum(embedding > 0)),
            "negative_count": int(np.sum(embedding < 0)),
            "zero_count": int(np.sum(embedding == 0)),
            "abs_mean": float(np.mean(abs_embedding)),
            "significant_dims": int(np.sum(abs_embedding > 0.1))
        }

        # Find activation clusters
        significant_threshold = 0.1
        significant_dims = np.where(abs_embedding > significant_threshold)[0]

        # Find clusters (dimensions that are close to each other)
        clusters = []
        if len(significant_dims) > 0:
            current_cluster = [int(significant_dims[0])]  # Convert to int

            for i in range(1, len(significant_dims)):
                if significant_dims[i] - significant_dims[i-1] <= 5:  # If dimensions are close
                    current_cluster.append(int(significant_dims[i]))  # Convert to int
                else:
                    if len(current_cluster) > 0:
                        clusters.append(current_cluster)
                    current_cluster = [int(significant_dims[i])]  # Convert to int

            if len(current_cluster) > 0:
                clusters.append(current_cluster)

        # Filter to meaningful clusters (more than 1 dimension)
        clusters = [c for c in clusters if len(c) > 1]

        # Format clusters for display
        cluster_info = []
        for i, cluster in enumerate(clusters):
            values = [float(embedding[dim]) for dim in cluster]  # Convert to float
            cluster_info.append({
                "id": i+1,
                "dimensions": [int(d) for d in cluster],  # Convert to int
                "start_dim": int(min(cluster)),
                "end_dim": int(max(cluster)),
                "size": int(len(cluster)),
                "avg_value": float(np.mean(values)),
                "max_value": float(np.max(values)),
                "max_dim": int(cluster[np.argmax(values)])
            })

        # Top dimensions by magnitude
        top_indices = sorted(range(len(embedding)), key=lambda i: abs(embedding[i]), reverse=True)[:10]
        top_dimensions = [{"dimension": int(idx), "value": float(embedding[idx])} for idx in top_indices]

        return {
            "metrics": metrics,
            "clusters": cluster_info,
            "top_dimensions": top_dimensions
        }
    except Exception as e:
        st.error(f"Error analyzing embedding: {e}")
        # Return a minimal valid structure in case of error
        return {
            "metrics": {
                "dimension_count": 0,
                "mean_value": 0.0,
                "std_dev": 0.0,
                "min_value": 0.0,
                "min_dimension": 0,
                "max_value": 0.0,
                "max_dimension": 0,
                "median_value": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "zero_count": 0,
                "abs_mean": 0.0,
                "significant_dims": 0
            },
            "clusters": [],
            "top_dimensions": []
        }

def create_report_pdf(embedding, analysis, claude_analysis, business_type, page_type):
    """Create a better-formatted PDF report of the analysis results with business and page type context"""
    try:
        # Create a buffer for the PDF
        buffer = BytesIO()

        # Create the PDF document with better margins
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )

        # Create custom styles
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading1_style = styles['Heading1']
        heading2_style = styles['Heading2']

        # Create a better normal style with more space
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            spaceBefore=6,
            spaceAfter=6,
            leading=14  # Line spacing
        )

        # Create a story (content)
        story = []

        # Add title with better spacing
        story.append(Paragraph("SEO Embedding Analysis Report", title_style))
        story.append(Spacer(1, 24))  # More space after title

        # Add date with better formatting
        from datetime import datetime
        date_style = ParagraphStyle(
            'DateStyle',
            parent=normal_style,
            fontName='Helvetica-Oblique',
            textColor=colors.darkblue
        )
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", date_style))
        story.append(Spacer(1, 12))  # Space after date

        # Add content categorization information
        category_style = ParagraphStyle(
            'CategoryStyle',
            parent=normal_style,
            fontName='Helvetica-Bold',
            textColor=colors.darkblue,
            fontSize=11
        )

        # Convert internal values to display values
        business_display = {
            "lead_generation": "Lead Generation/Service",
            "ecommerce": "E-commerce",
            "saas": "SaaS/Tech",
            "educational": "Educational/Informational",
            "local_business": "Local Business"
        }.get(business_type, business_type.replace("_", " ").title())

        page_display = page_type.replace("_", " ").title()

        story.append(Paragraph(f"Business Type: {business_display}", category_style))
        story.append(Paragraph(f"Page Type: {page_display}", category_style))
        
        # Add keyword information if available
        if st.session_state.target_keywords.strip():
            keywords_list = [kw.strip() for kw in st.session_state.target_keywords.split('\n') if kw.strip()]
            keywords_text = ", ".join(keywords_list[:5])
            if len(keywords_list) > 5:
                keywords_text += f" (+{len(keywords_list) - 5} more)"
            story.append(Paragraph(f"Target Keywords: {keywords_text}", category_style))
        
        story.append(Spacer(1, 24))  # More space after categories

        # Add metrics section with better formatting
        story.append(Paragraph("Key Metrics", heading1_style))
        story.append(Spacer(1, 12))

        # Format metrics as a table for better appearance
        metrics = analysis["metrics"]
        metrics_data = [
            ["Metric", "Value"],
            ["Dimensions", f"{metrics['dimension_count']}"],
            ["Mean Value", f"{metrics['mean_value']:.6f}"],
            ["Standard Deviation", f"{metrics['std_dev']:.6f}"],
            ["Min Value", f"{metrics['min_value']:.6f} (dim {metrics['min_dimension']})"],
            ["Max Value", f"{metrics['max_value']:.6f} (dim {metrics['max_dimension']})"],
            ["Positive Values", f"{metrics['positive_count']} ({metrics['positive_count']/metrics['dimension_count']*100:.2f}%)"],
            ["Negative Values", f"{metrics['negative_count']} ({metrics['negative_count']/metrics['dimension_count']*100:.2f}%)"],
            ["Significant Dimensions", f"{metrics['significant_dims']} (>0.1)"]
        ]

        # Create a table with the metrics
        metrics_table = Table(metrics_data, colWidths=[2*inch, 3.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 24))

        # Create visualizations page with better layout
        story.append(PageBreak())
        story.append(Paragraph("Embedding Visualizations", heading1_style))
        story.append(Spacer(1, 16))

        # Function to add an image with a better caption
        def add_figure_with_caption(story, img_data, caption):
            # Add caption with better styling
            caption_style = ParagraphStyle(
                'CaptionStyle',
                parent=heading2_style,
                textColor=colors.darkblue,
                spaceAfter=8
            )
            story.append(Paragraph(caption, caption_style))

            # Create image with better size settings
            img = Image(img_data, width=7*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 16))

        # Create plots with better resolution and save them as images
        # 1. Embedding Overview
        fig_overview = plot_embedding_overview(embedding)
        fig_overview.set_dpi(150)  # Higher resolution
        fig_overview.tight_layout()  # Better layout
        overview_img = BytesIO()
        fig_overview.savefig(overview_img, format='png', bbox_inches='tight')
        plt.close(fig_overview)
        overview_img.seek(0)

        # 2. Top Dimensions
        fig_top = plot_top_dimensions(embedding)
        fig_top.set_dpi(150)
        fig_top.tight_layout()
        top_img = BytesIO()
        fig_top.savefig(top_img, format='png', bbox_inches='tight')
        plt.close(fig_top)
        top_img.seek(0)

        # 3. Activation Histogram
        fig_hist = plot_activation_histogram(embedding)
        fig_hist.set_dpi(150)
        fig_hist.tight_layout()
        hist_img = BytesIO()
        fig_hist.savefig(hist_img, format='png', bbox_inches='tight')
        plt.close(fig_hist)
        hist_img.seek(0)

        # Add the first three visualizations
        add_figure_with_caption(story, overview_img, "Embedding Overview")
        add_figure_with_caption(story, top_img, "Top 20 Dimensions by Magnitude")
        add_figure_with_caption(story, hist_img, "Distribution of Embedding Values")

        # Start a new page for the next visualizations
        story.append(PageBreak())

        # 4. Dimension Clusters
        fig_clusters = plot_dimension_clusters(embedding)
        fig_clusters.set_dpi(150)
        fig_clusters.tight_layout()
        clusters_img = BytesIO()
        fig_clusters.savefig(clusters_img, format='png', bbox_inches='tight')
        plt.close(fig_clusters)
        clusters_img.seek(0)

        # 5. PCA Visualization
        fig_pca = plot_pca(embedding)
        fig_pca.set_dpi(150)
        fig_pca.tight_layout()
        pca_img = BytesIO()
        fig_pca.savefig(pca_img, format='png', bbox_inches='tight')
        plt.close(fig_pca)
        pca_img.seek(0)

        # Add the remaining visualizations
        add_figure_with_caption(story, clusters_img, "Dimension Clusters Heatmap")
        add_figure_with_caption(story, pca_img, "PCA Visualization of Embedding Segments")

        # Add Claude Analysis section with better formatting
        story.append(PageBreak())
        story.append(Paragraph("Comprehensive Analysis", heading1_style))
        story.append(Spacer(1, 16))

        # Add categorization banner for context
        context_style = ParagraphStyle(
            'ContextStyle',
            parent=normal_style,
            fontName='Helvetica-Bold',
            textColor=colors.white,
            fontSize=10,
            alignment=1,  # Center alignment
            backColor=colors.darkblue
        )

        context_text = f"Analysis tailored for: {business_display} | {page_display}"
        if st.session_state.target_keywords.strip():
            keywords_list = [kw.strip() for kw in st.session_state.target_keywords.split('\n') if kw.strip()]
            context_text += f" | Keywords: {len(keywords_list)} targeted"

        story.append(Paragraph(context_text, context_style))
        story.append(Spacer(1, 12))

        # Process Claude analysis text - convert from HTML/markdown to plain text
        # Remove HTML tags
        analysis_text = re.sub(r'<[^>]*>', '', claude_analysis)

        # Function to process headings better
        def process_heading(text, style):
            text = text.strip()
            if text.startswith('##'):
                return Paragraph(text.replace('##', '').strip(), heading1_style)
            elif text.startswith('#'):
                return Paragraph(text.replace('#', '').strip(), heading2_style)
            elif text.strip() and text.strip()[0].isupper():  # Likely a section title without # marks
                first_line = text.split("\n")[0].strip()
                if len(first_line) < 50 and first_line.isupper():  # It's probably a heading
                    return Paragraph(first_line, heading2_style)
            # Normal paragraph
            return Paragraph(text, normal_style)

        # Split by paragraphs and process each one
        paragraphs = analysis_text.split('\n\n')

        for paragraph in paragraphs:
            if paragraph.strip():
                element = process_heading(paragraph, normal_style)
                story.append(element)
                if isinstance(element, Paragraph) and element.style == heading1_style:
                    story.append(Spacer(1, 12))  # More space after main headings

        # Build the PDF
        doc.build(story)

        # Get the PDF value
        pdf_value = buffer.getvalue()
        buffer.close()

        return pdf_value

    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

# Main content area
def main():
    st.title("SEO Embedding Analysis Tool")

    # Content input options
    st.header("Content Input")

    input_method = st.radio("Choose input method:", ("Paste Content", "Fetch from URL"), key="input_method_radio")

    # Use a placeholder for content that reflects the current input method
    content_placeholder = "Enter the content you want to analyze..." if input_method == "Paste Content" else "Content will be fetched from the URL..."

    if input_method == "Paste Content":
        # Text area for pasting content
        # Value is read directly from session state on rerun
        st.session_state.content = st.text_area("Paste your content here:", height=300,
                              placeholder=content_placeholder,
                              value=st.session_state.get('content', ''), key="pasted_content_area")
        st.session_state.url = "" # Clear URL state when pasting content
        # Button for analyzing pasted content
        analyze_pasted_button = st.button("Analyze Pasted Content")
        analyze_fetch_button = False # Ensure fetch button state doesn't interfere

    else: # Fetch from URL
        # Text input for URL
        url = st.text_input("Enter the URL:", value=st.session_state.get('url', ''),
                           placeholder="e.g., https://www.example.com/your-page")
        # Update session state URL
        st.session_state.url = url
        # Clear pasted content state initially for URL mode on radio button change
        # This is handled implicitly by the input_method check below

        # Button to trigger fetch and then analysis
        analyze_fetch_button = st.button("Fetch and Analyze URL")
        analyze_pasted_button = False # Ensure paste button state doesn't interfere

        # Logic to fetch content when the fetch button is clicked
        if analyze_fetch_button and url:
             with st.spinner(f"Fetching content from {url}..."):
                fetched_content = fetch_content_from_url(url)
                if fetched_content:
                    st.session_state.content = fetched_content # Store fetched content
                    st.success("Content fetched successfully! Running analysis...")
                    st.session_state.fetch_button_clicked = True # Indicate successful fetch for the analysis trigger
                else:
                     st.error("Failed to fetch content. Please check the URL or try pasting.")
                     st.session_state.content = "" # Clear content on failure
                     st.session_state.url = "" # Clear URL on failure
                     st.session_state.fetch_button_clicked = False # Reset state on failure if fetch fails
             # Trigger a rerun after fetch attempt to update the UI and potentially start analysis
             raise RerunException(RerunData()) # Use the correct rerun method with RerunData

        elif analyze_fetch_button and not url:
             st.warning("Please enter a URL.")
             st.session_state.fetch_button_clicked = False # Reset state if button clicked with no URL

    # Display fetched content in a text area for review if available and in URL mode
    # This text area also allows editing the fetched content before analysis if needed
    if st.session_state.content and input_method == "Fetch from URL":
         st.subheader("Fetched Content (Review)")
         # Use a unique key for the text area in URL mode to maintain its state separately
         st.session_state.content = st.text_area("Review and edit fetched content:", value=st.session_state.content, height=300, key="fetched_content_review_area")

    # Enhanced content categorization section with keyword targeting
    st.write("### Content Categorization & SEO Targeting")
    st.write("Categorize your content and specify target keywords for more precise SEO analysis:")

    # Create three columns for the dropdown menus and keyword input
    cat_col1, cat_col2, cat_col3 = st.columns([1, 1, 1.2])

    with cat_col1:
        business_type = st.selectbox(
            "Business Type:",
            options=["Lead Generation/Service", "E-commerce", "SaaS/Tech", "Educational/Informational", "Local Business"],
            index=["Lead Generation/Service", "E-commerce", "SaaS/Tech", "Educational/Informational", "Local Business"].index(
                "Lead Generation/Service" if st.session_state.business_type == "lead_generation" else
                "E-commerce" if st.session_state.business_type == "ecommerce" else
                "SaaS/Tech" if st.session_state.business_type == "saas" else
                "Educational/Informational" if st.session_state.business_type == "educational" else
                "Local Business"
            ),
            help="Select the business model that best matches your content",
            key="business_type_selectbox"
        )

        # Convert display values to internal values
        st.session_state.business_type = (
            "lead_generation" if business_type == "Lead Generation/Service" else
            "ecommerce" if business_type == "E-commerce" else
            "saas" if business_type == "SaaS/Tech" else
            "educational" if business_type == "Educational/Informational" else
            "local_business"
        )

    # Dynamically change page type options based on business type
    with cat_col2:
        if st.session_state.business_type == "lead_generation":
            page_options = ["Landing Page", "Service Page", "Blog Post", "About Us", "Contact Page"]
        elif st.session_state.business_type == "ecommerce":
            page_options = ["Product Page", "Category Page", "Homepage", "Blog Post", "Checkout Page"]
        elif st.session_state.business_type == "saas":
            page_options = ["Feature Page", "Pricing Page", "Homepage", "Blog Post", "Documentation"]
        elif st.session_state.business_type == "educational":
            page_options = ["Course Page", "Resource Page", "Blog Post", "Homepage", "About Us"]
        else:  # local_business
            page_options = ["Homepage", "Service Page", "Location Page", "About Us", "Contact Page"]

        # Get current page type and find its index in the new options, defaulting to 0 if not found
        current_page_display = next((p for p in page_options if p.lower().replace(" ", "_") == st.session_state.page_type), page_options[0])
        current_page_index = page_options.index(current_page_display) if current_page_display in page_options else 0

        page_type = st.selectbox(
            "Page Type:",
            options=page_options,
            index=current_page_index,
            help="Select the type of page you're analyzing",
            key="page_type_selectbox"
        )

        # Convert display values to internal values
        st.session_state.page_type = page_type.lower().replace(" ", "_")

    # New keyword targeting input
    with cat_col3:
        target_keywords = st.text_area(
            "Target Keywords:",
            value=st.session_state.target_keywords,
            height=100,
            placeholder="Enter your target keywords, one per line:\n\ndigital marketing services\nSEO optimization\nbrand strategy\nlocal SEO",
            help="Enter the keywords you want to rank for. Put each keyword on a new line. Include both primary and secondary keywords.",
            key="target_keywords_input"
        )
        
        st.session_state.target_keywords = target_keywords
        
        # Show keyword count and validation
        if target_keywords.strip():
            keywords_list = [kw.strip() for kw in target_keywords.split('\n') if kw.strip()]
            st.caption(f"Keywords entered: {len(keywords_list)}")
            
            # Show a preview of keywords in an expander
            with st.expander("Preview Keywords", expanded=False):
                for i, kw in enumerate(keywords_list[:10], 1):  # Show first 10
                    st.write(f"{i}. {kw}")
                if len(keywords_list) > 10:
                    st.write(f"... and {len(keywords_list) - 10} more")
        else:
            st.caption("Add keywords for targeted analysis")

    # Add keyword strategy tips
    with st.expander("Keyword Strategy Tips", expanded=False):
        st.markdown("""
        **Effective Keyword Targeting:**
        - **Primary Keywords**: 1-3 main keywords you want to rank for
        - **Secondary Keywords**: 5-10 related terms and long-tail variations
        - **Local Keywords**: Include location-based terms for local businesses
        - **Intent-based Keywords**: Match keywords to user search intent
        
        **Examples by Business Type:**
        - **Lead Gen**: "digital marketing agency", "PPC management services"
        - **E-commerce**: "wireless headphones", "bluetooth earbuds under $100"
        - **SaaS**: "project management software", "team collaboration tools"
        - **Educational**: "online courses", "digital marketing certification"
        - **Local**: "dentist near me", "plumber in [city]"
        """)

    # --- Analysis Trigger Logic ---
    # Determine if analysis should be triggered
    # Trigger analysis if either the paste button is clicked and content is available,
    # OR if the fetch button was clicked in the previous run and content is now successfully loaded in session state.
    trigger_analysis = False

    # Case 1: Paste Content and Analyze button clicked
    if input_method == "Paste Content" and analyze_pasted_button and st.session_state.content:
         trigger_analysis = True

    # Case 2: Fetch from URL button clicked and fetch was successful (state updated in previous rerun)
    # We check fetch_button_clicked which was set True in the previous rerun upon successful fetch
    elif input_method == "Fetch from URL" and st.session_state.fetch_button_clicked and st.session_state.content:
         trigger_analysis = True
         # Reset the fetch_button_clicked state immediately after triggering analysis
         st.session_state.fetch_button_clicked = False

    # --- Analysis Execution ---
    # This block runs if the trigger_analysis flag is True based on the conditions above
    if trigger_analysis:

        # Reset PDF data if a new analysis is starting
        st.session_state.pdf_data = None
        st.session_state.pdf_generated = False
        st.session_state.claude_analysis = None # Clear previous analysis results

        # Check if API keys are provided before proceeding with analysis
        if not st.session_state.google_api_key or not st.session_state.anthropic_api_key:
            st.error("Please provide both Google API and Anthropic API keys in the sidebar.")
            # Reset analysis state if keys are missing to prevent displaying old results
            st.session_state.embedding = None
            st.session_state.analysis = None
            st.session_state.claude_analysis = None
            # No rerun needed here, the error message is sufficient to stop
            return

        with st.spinner("Analyzing content with keyword targeting... This may take a minute."):

            # Get embedding with progress indicator
            progress_text = st.empty()
            progress_text.text("Generating embedding from Google Gemini API...")
            st.session_state.embedding = get_embedding(st.session_state.content)

            # Generate analysis
            progress_text.text("Analyzing embedding patterns...")
            st.session_state.analysis = analyze_embedding(st.session_state.embedding)

            # Get Claude analysis with business, page type, and keyword context
            progress_text.text("Getting comprehensive SEO and keyword analysis from Claude...")
            # Increase content sent to Claude to include keyword analysis section
            claude_content_snippet = st.session_state.content[:7000]
            st.session_state.claude_analysis = analyze_with_claude(
                st.session_state.embedding,
                claude_content_snippet,
                st.session_state.business_type,
                st.session_state.page_type
            )
            progress_text.empty()

            # Display results only after successful analysis
            if st.session_state.claude_analysis and "Error getting analysis" not in st.session_state.claude_analysis:
                 st.success("Analysis complete with keyword targeting insights!")
                 # Trigger a rerun to display the results tabs
                 raise RerunException(RerunData())
            else:
                 st.error("Analysis failed. Please check your API keys and try again.")

    # Only display the results if we have a completed analysis in the session state
    # Check for both embedding and claude_analysis to ensure analysis is complete and not an error message
    if st.session_state.embedding is not None and st.session_state.claude_analysis is not None and "Error getting analysis" not in st.session_state.claude_analysis:
        # Create tabs for different sections with keyword insights
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Visualizations", "Metrics", "Keywords", "Clusters", "Analysis Report"])

        # Tab 1: Visualizations
        with tab1:
            st.subheader("Embedding Overview")
            fig1 = plot_embedding_overview(st.session_state.embedding)
            st.pyplot(fig1)
            plt.close(fig1)  # Release memory

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top Dimensions")
                fig2 = plot_top_dimensions(st.session_state.embedding)
                st.pyplot(fig2)
                plt.close(fig2)  # Release memory

            with col2:
                st.subheader("Activation Distribution")
                fig3 = plot_activation_histogram(st.session_state.embedding)
                st.pyplot(fig3)
                plt.close(fig3)  # Release memory

            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Dimension Clusters")
                fig4 = plot_dimension_clusters(st.session_state.embedding)
                st.pyplot(fig4)
                plt.close(fig4)  # Release memory

            with col4:
                st.subheader("PCA Visualization")
                fig5 = plot_pca(st.session_state.embedding)
                st.pyplot(fig5)
                plt.close(fig5)  # Release memory

        # Tab 2: Metrics
        with tab2:
            st.subheader("Key Metrics")
            metrics = st.session_state.analysis["metrics"]

            # Create a 3-column layout for metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Dimensions", metrics["dimension_count"])
                st.metric("Mean Value", f"{metrics['mean_value']:.6f}")
                st.metric("Standard Deviation", f"{metrics['std_dev']:.6f}")

            with col2:
                st.metric("Min Value", f"{metrics['min_value']:.6f} (dim {metrics['min_dimension']})")
                st.metric("Max Value", f"{metrics['max_value']:.6f} (dim {metrics['max_dimension']})")
                st.metric("Median Value", f"{metrics['median_value']:.6f}")

            with col3:
                st.metric("Positive Values", f"{metrics['positive_count']} ({metrics['positive_count']/metrics['dimension_count']*100:.2f}%)")
                st.metric("Negative Values", f"{metrics['negative_count']} ({metrics['negative_count']/metrics['dimension_count']*100:.2f}%)")
                st.metric("Significant Dimensions", f"{metrics['significant_dims']} (>0.1)")

        # Tab 3: NEW Keyword Analysis Tab
        with tab3:
            st.subheader("Keyword Targeting Analysis")
            
            if st.session_state.target_keywords.strip():
                keywords_list = [kw.strip() for kw in st.session_state.target_keywords.split('\n') if kw.strip()]
                
                # Keyword overview
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Keywords", len(keywords_list))
                with col2:
                    st.metric("Primary Keywords", min(3, len(keywords_list)))
                with col3:
                    st.metric("Long-tail Keywords", max(0, len(keywords_list) - 3))
                
                # Display keywords with analysis if content contains keyword analysis
                if "=== KEYWORD TARGETING ANALYSIS ===" in st.session_state.content:
                    st.subheader("Keyword Performance Summary")
                    
                    # Parse keyword analysis from content (this is a simplified approach)
                    # In a production app, you'd want to store this data separately
                    content_lines = st.session_state.content.split('\n')
                    keyword_section_start = False
                    current_keyword = None
                    
                    for line in content_lines:
                        if "=== KEYWORD TARGETING ANALYSIS ===" in line:
                            keyword_section_start = True
                            continue
                        elif "=== CONTENT STATISTICS ===" in line:
                            break
                        elif keyword_section_start and line.startswith("KEYWORD:"):
                            current_keyword = line.replace("KEYWORD:", "").strip()
                            with st.expander(f"Analysis for: {current_keyword}", expanded=False):
                                # Extract keyword data from the following lines
                                # This is a simplified version - you might want to enhance this
                                st.write("**Keyword Analysis:**")
                                st.write("• Detailed analysis available in the full report")
                                st.write("• Check the Analysis Report tab for complete insights")
                else:
                    st.info("Keyword analysis will be included in the next content fetch or analysis.")
                
                # Show keyword list
                st.subheader("Target Keywords List")
                for i, keyword in enumerate(keywords_list, 1):
                    keyword_type = "Primary" if i <= 3 else "Secondary"
                    st.write(f"{i}. **{keyword}** _{keyword_type}_")
                
            else:
                st.info("**No target keywords specified**")
                st.write("Add target keywords in the Content Categorization section to see detailed keyword analysis including:")
                st.write("• Keyword density analysis")
                st.write("• SEO element integration")
                st.write("• Semantic alignment scores") 
                st.write("• Optimization recommendations")
                
                with st.expander("Keyword Research Tips", expanded=False):
                    st.markdown("""
                    **How to choose effective target keywords:**
                    
                    1. **Primary Keywords (1-3)**: Main terms you want to rank for
                    2. **Secondary Keywords (5-10)**: Supporting and long-tail variations
                    3. **Intent Matching**: Choose keywords that match your page purpose
                    4. **Competition Analysis**: Balance search volume with ranking difficulty
                    
                    **Keyword Research Tools:**
                    - Google Keyword Planner
                    - SEMrush / Ahrefs
                    - Google Search Console
                    - Answer The Public
                    """)

        # Tab 4: Clusters
        with tab4:
            st.subheader("Dimension Clusters")

            if not st.session_state.analysis["clusters"]:
                st.info("No significant dimension clusters detected.")
            else:
                for cluster in st.session_state.analysis["clusters"]:
                    with st.expander(f"Cluster #{cluster['id']}: Dimensions {cluster['start_dim']}-{cluster['end_dim']}"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Size", f"{cluster['size']} dimensions")
                        col2.metric("Avg Value", f"{cluster['avg_value']:.6f}")
                        col3.metric("Max Value", f"{cluster['max_value']:.6f} (dim {cluster['max_dim']})")

        # Tab 5: Enhanced Claude Analysis with keyword context
        with tab5:
            # Show content categorization info with keywords
            business_display = {
                "lead_generation": "Lead Generation/Service",
                "ecommerce": "E-commerce",
                "saas": "SaaS/Tech",
                "educational": "Educational/Informational",
                "local_business": "Local Business"
            }.get(st.session_state.business_type, st.session_state.business_type.replace("_", " ").title())

            page_display = st.session_state.page_type.replace("_", " ").title()
            
            # Enhanced context display with keywords
            keywords_display = ""
            if st.session_state.target_keywords.strip():
                keywords_list = [kw.strip() for kw in st.session_state.target_keywords.split('\n') if kw.strip()]
                keywords_display = f"<p style=\"margin: 0;\"><strong>Target Keywords:</strong> {', '.join(keywords_list[:5])}"
                if len(keywords_list) > 5:
                    keywords_display += f" (+{len(keywords_list) - 5} more)"
                keywords_display += "</p>"
            else:
                keywords_display = "<p style=\"margin: 0;\"><strong>Target Keywords:</strong> <em>None specified</em></p>"

            st.markdown(f"""
            <div style="background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <p style="margin: 0; font-weight: bold; color: #1f4e79;">Analysis Context:</p>
                <p style="margin: 0;"><strong>Business Type:</strong> {business_display}</p>
                <p style="margin: 0;"><strong>Page Type:</strong> {page_display}</p>
                {keywords_display}
            </div>
            """, unsafe_allow_html=True)

            st.subheader("Comprehensive SEO & Keyword Analysis Report")

            # Display the analysis content
            st.markdown(st.session_state.claude_analysis, unsafe_allow_html=True)

            # Enhanced download options
            st.write("---")

            with st.container():
                st.markdown("""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
                <h3 style="margin-top: 0;">Download Options</h3>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                # Text download button
                with col1:
                    st.download_button(
                        label="Download as Text",
                        data=st.session_state.claude_analysis,
                        file_name="seo_keyword_analysis_report.txt",
                        mime="text/plain",
                        help="Download the analysis as a plain text file",
                        key="download_text_button"
                    )

                # PDF generation button and download
                with col2:
                    if st.session_state.pdf_data is None:
                        if st.button("Generate PDF Report", help="Create a PDF with visualizations and analysis", key="generate_pdf_button_tab"):
                            with st.spinner("Generating comprehensive PDF report with keyword analysis... This may take up to a minute."):
                                try:
                                    pdf_bytes = create_report_pdf(
                                        st.session_state.embedding,
                                        st.session_state.analysis,
                                        st.session_state.claude_analysis,
                                        st.session_state.business_type,
                                        st.session_state.page_type
                                    )
                                    st.session_state.pdf_data = pdf_bytes
                                    st.session_state.pdf_generated = True
                                    st.success("PDF report with keyword insights generated successfully!")
                                    # Trigger a rerun to show the download button immediately
                                    raise RerunException(RerunData())
                                except Exception as e:
                                    st.error(f"Error generating PDF: {str(e)}")
                    else:
                         # Show PDF download button if PDF has been generated
                         b64_pdf = base64.b64encode(st.session_state.pdf_data).decode('utf-8')
                         download_link = f'<a href="data:application/pdf;base64,{b64_pdf}" download="seo_keyword_analysis_report.pdf" class="button" style="display: inline-block; padding: 12px 20px; background-color: #0c6b58; color: white; text-decoration: none; font-weight: bold; border-radius: 4px; text-align: center; margin: 10px 0; width: 100%;">DOWNLOAD COMPLETE PDF REPORT</a>'
                         st.markdown(download_link, unsafe_allow_html=True)

                         if st.button("Generate New PDF", key="discard_pdf_button_tab"):
                            st.session_state.pdf_data = None
                            st.session_state.pdf_generated = False
                            # Trigger a rerun to update the button state
                            raise RerunException(RerunData())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        # Add a basic content area to ensure something is visible
        st.write("## SEO Embedding Analysis Tool")
        st.write("This tool analyzes content semantics using embeddings."), r'<strong>\1</strong>', analysis_text, flags=re.MULTILINE)
    return formatted_text

def plot_embedding_overview(embedding):
    """Create overview plot of embedding values"""
    try:
        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        ax.plot(range(len(embedding)), embedding)
        ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax.set_title('Embedding Values Across All 3k Dimensions')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        return fig
    except Exception as e:
        st.error(f"Error plotting embedding overview: {e}")
        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center')
        return fig

def plot_top_dimensions(embedding):
    """Plot top dimensions by magnitude"""
    try:
        # Get indices of top 20 dimensions by magnitude
        top_indices = sorted(range(len(embedding)), key=lambda i: abs(embedding[i]), reverse=True)[:20]
        top_values = [embedding[i] for i in top_indices]

        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        colors = ['blue' if v >= 0 else 'red' for v in top_values]
        ax.bar(range(len(top_indices)), top_values, color=colors)
        ax.set_xticks(range(len(top_indices)))
        ax.set_xticklabels(top_indices, rotation=45)
        ax.set_title('Top 20 Dimensions by Magnitude')
        ax.set_xlabel('Dimension Index')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        return fig
    except Exception as e:
        st.error(f"Error plotting top dimensions: {e}")
        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center')
        return fig

def plot_dimension_clusters(embedding):
    """Plot dimension clusters heatmap"""
    try:
        # Reshape embedding to highlight patterns
        embedding_reshaped = np.array(embedding).reshape(64, 48)

        fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
        # Create a custom colormap from blue to white to red
        cmap = LinearSegmentedColormap.from_list('BrBG', ['blue', 'white', 'red'], N=256)
        im = ax.imshow(embedding_reshaped, cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax, label='Activation Value')
        ax.set_title('Embedding Clusters Heatmap (Reshaped to 64x48)')
        ax.set_xlabel('Dimension Group')
        ax.set_ylabel('Dimension Group')
        return fig
    except Exception as e:
        st.error(f"Error plotting dimension clusters: {e}")
        fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center')
        return fig

def plot_pca(embedding):
    """Plot PCA visualization of embedding dimensions"""
    try:
        # Create a 2D array where each row is a segment of the original embedding
        segment_size = 256
        num_segments = len(embedding) // segment_size
        data_matrix = np.zeros((num_segments, segment_size))

        # Fill the matrix with segments
        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size
            data_matrix[i] = embedding[start:end]

        # Apply PCA
        fig = plt.figure(figsize=(10, 8))
        if num_segments > 1:
            pca = PCA(n_components=2)
            pca_results = pca.fit_transform(data_matrix)

            plt.scatter(pca_results[:, 0], pca_results[:, 1])

            # Label each point with its segment range
            for i in range(num_segments):
                start = i * segment_size
                end = start + segment_size - 1
                plt.annotate(f"{start}-{end}",
                            (pca_results[i, 0], pca_results[i, 1]),
                            fontsize=8)

            plt.title('PCA of Embedding Segments')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.grid(True, alpha=0.3)
        else:
            # If we don't have enough segments, create a simpler visualization
            plt.text(0.5, 0.5, "Not enough segments for PCA visualization",
                    ha='center', va='center', fontsize=12)
            plt.axis('off')

        return fig
    except Exception as e:
        st.error(f"Error plotting PCA: {e}")
        fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center')
        return fig

def plot_activation_histogram(embedding):
    """Plot histogram of embedding activation values"""
    try:
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        ax.hist(embedding, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        ax.set_title('Distribution of Embedding Values')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        return fig
    except Exception as e:
        st.error(f"Error plotting activation histogram: {e}")
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        ax.text(0.5, 0.5, f"Error generating plot: {str(e)}",
                horizontalalignment='center', verticalalignment='center')
        return fig

def analyze_embedding(embedding):
    """Analyze embedding for key metrics"""
    try:
        embedding = np.array(embedding)  # Convert to numpy array for easier processing
        abs_embedding = np.abs(embedding)

        # Calculate key metrics - CONVERT NUMPY TYPES TO PYTHON NATIVE TYPES
        metrics = {
            "dimension_count": int(len(embedding)),
            "mean_value": float(np.mean(embedding)),
            "std_dev": float(np.std(embedding)),
            "min_value": float(np.min(embedding)),
            "min_dimension": int(np.argmin(embedding)),
            "max_value": float(np.max(embedding)),
            "max_dimension": int(np.argmax(embedding)),
            "median_value": float(np.median(embedding)),
            "positive_count": int(np.sum(embedding > 0)),
            "negative_count": int(np.sum(embedding < 0)),
            "zero_count": int(np.sum(embedding == 0)),
            "abs_mean": float(np.mean(abs_embedding)),
            "significant_dims": int(np.sum(abs_embedding > 0.1))
        }

        # Find activation clusters
        significant_threshold = 0.1
        significant_dims = np.where(abs_embedding > significant_threshold)[0]

        # Find clusters (dimensions that are close to each other)
        clusters = []
        if len(significant_dims) > 0:
            current_cluster = [int(significant_dims[0])]  # Convert to int

            for i in range(1, len(significant_dims)):
                if significant_dims[i] - significant_dims[i-1] <= 5:  # If dimensions are close
                    current_cluster.append(int(significant_dims[i]))  # Convert to int
                else:
                    if len(current_cluster) > 0:
                        clusters.append(current_cluster)
                    current_cluster = [int(significant_dims[i])]  # Convert to int

            if len(current_cluster) > 0:
                clusters.append(current_cluster)

        # Filter to meaningful clusters (more than 1 dimension)
        clusters = [c for c in clusters if len(c) > 1]

        # Format clusters for display
        cluster_info = []
        for i, cluster in enumerate(clusters):
            values = [float(embedding[dim]) for dim in cluster]  # Convert to float
            cluster_info.append({
                "id": i+1,
                "dimensions": [int(d) for d in cluster],  # Convert to int
                "start_dim": int(min(cluster)),
                "end_dim": int(max(cluster)),
                "size": int(len(cluster)),
                "avg_value": float(np.mean(values)),
                "max_value": float(np.max(values)),
                "max_dim": int(cluster[np.argmax(values)])
            })

        # Top dimensions by magnitude
        top_indices = sorted(range(len(embedding)), key=lambda i: abs(embedding[i]), reverse=True)[:10]
        top_dimensions = [{"dimension": int(idx), "value": float(embedding[idx])} for idx in top_indices]

        return {
            "metrics": metrics,
            "clusters": cluster_info,
            "top_dimensions": top_dimensions
        }
    except Exception as e:
        st.error(f"Error analyzing embedding: {e}")
        # Return a minimal valid structure in case of error
        return {
            "metrics": {
                "dimension_count": 0,
                "mean_value": 0.0,
                "std_dev": 0.0,
                "min_value": 0.0,
                "min_dimension": 0,
                "max_value": 0.0,
                "max_dimension": 0,
                "median_value": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "zero_count": 0,
                "abs_mean": 0.0,
                "significant_dims": 0
            },
            "clusters": [],
            "top_dimensions": []
        }

def create_report_pdf(embedding, analysis, claude_analysis, business_type, page_type):
    """Create a better-formatted PDF report of the analysis results with business and page type context"""
    try:
        # Create a buffer for the PDF
        buffer = BytesIO()

        # Create the PDF document with better margins
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )

        # Create custom styles
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading1_style = styles['Heading1']
        heading2_style = styles['Heading2']

        # Create a better normal style with more space
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            spaceBefore=6,
            spaceAfter=6,
            leading=14  # Line spacing
        )

        # Create a story (content)
        story = []

        # Add title with better spacing
        story.append(Paragraph("SEO Embedding Analysis Report", title_style))
        story.append(Spacer(1, 24))  # More space after title

        # Add date with better formatting
        from datetime import datetime
        date_style = ParagraphStyle(
            'DateStyle',
            parent=normal_style,
            fontName='Helvetica-Oblique',
            textColor=colors.darkblue
        )
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", date_style))
        story.append(Spacer(1, 12))  # Space after date

        # Add content categorization information
        category_style = ParagraphStyle(
            'CategoryStyle',
            parent=normal_style,
            fontName='Helvetica-Bold',
            textColor=colors.darkblue,
            fontSize=11
        )

        # Convert internal values to display values
        business_display = {
            "lead_generation": "Lead Generation/Service",
            "ecommerce": "E-commerce",
            "saas": "SaaS/Tech",
            "educational": "Educational/Informational",
            "local_business": "Local Business"
        }.get(business_type, business_type.replace("_", " ").title())

        page_display = page_type.replace("_", " ").title()

        story.append(Paragraph(f"Business Type: {business_display}", category_style))
        story.append(Paragraph(f"Page Type: {page_display}", category_style))
        
        # Add keyword information if available
        if st.session_state.target_keywords.strip():
            keywords_list = [kw.strip() for kw in st.session_state.target_keywords.split('\n') if kw.strip()]
            keywords_text = ", ".join(keywords_list[:5])
            if len(keywords_list) > 5:
                keywords_text += f" (+{len(keywords_list) - 5} more)"
            story.append(Paragraph(f"Target Keywords: {keywords_text}", category_style))
        
        story.append(Spacer(1, 24))  # More space after categories

        # Add metrics section with better formatting
        story.append(Paragraph("Key Metrics", heading1_style))
        story.append(Spacer(1, 12))

        # Format metrics as a table for better appearance
        metrics = analysis["metrics"]
        metrics_data = [
            ["Metric", "Value"],
            ["Dimensions", f"{metrics['dimension_count']}"],
            ["Mean Value", f"{metrics['mean_value']:.6f}"],
            ["Standard Deviation", f"{metrics['std_dev']:.6f}"],
            ["Min Value", f"{metrics['min_value']:.6f} (dim {metrics['min_dimension']})"],
            ["Max Value", f"{metrics['max_value']:.6f} (dim {metrics['max_dimension']})"],
            ["Positive Values", f"{metrics['positive_count']} ({metrics['positive_count']/metrics['dimension_count']*100:.2f}%)"],
            ["Negative Values", f"{metrics['negative_count']} ({metrics['negative_count']/metrics['dimension_count']*100:.2f}%)"],
            ["Significant Dimensions", f"{metrics['significant_dims']} (>0.1)"]
        ]

        # Create a table with the metrics
        metrics_table = Table(metrics_data, colWidths=[2*inch, 3.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 24))

        # Create visualizations page with better layout
        story.append(PageBreak())
        story.append(Paragraph("Embedding Visualizations", heading1_style))
        story.append(Spacer(1, 16))

        # Function to add an image with a better caption
        def add_figure_with_caption(story, img_data, caption):
            # Add caption with better styling
            caption_style = ParagraphStyle(
                'CaptionStyle',
                parent=heading2_style,
                textColor=colors.darkblue,
                spaceAfter=8
            )
            story.append(Paragraph(caption, caption_style))

            # Create image with better size settings
            img = Image(img_data, width=7*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 16))

        # Create plots with better resolution and save them as images
        # 1. Embedding Overview
        fig_overview = plot_embedding_overview(embedding)
        fig_overview.set_dpi(150)  # Higher resolution
        fig_overview.tight_layout()  # Better layout
        overview_img = BytesIO()
        fig_overview.savefig(overview_img, format='png', bbox_inches='tight')
        plt.close(fig_overview)
        overview_img.seek(0)

        # 2. Top Dimensions
        fig_top = plot_top_dimensions(embedding)
        fig_top.set_dpi(150)
        fig_top.tight_layout()
        top_img = BytesIO()
        fig_top.savefig(top_img, format='png', bbox_inches='tight')
        plt.close(fig_top)
        top_img.seek(0)

        # 3. Activation Histogram
        fig_hist = plot_activation_histogram(embedding)
        fig_hist.set_dpi(150)
        fig_hist.tight_layout()
        hist_img = BytesIO()
        fig_hist.savefig(hist_img, format='png', bbox_inches='tight')
        plt.close(fig_hist)
        hist_img.seek(0)

        # Add the first three visualizations
        add_figure_with_caption(story, overview_img, "Embedding Overview")
        add_figure_with_caption(story, top_img, "Top 20 Dimensions by Magnitude")
        add_figure_with_caption(story, hist_img, "Distribution of Embedding Values")

        # Start a new page for the next visualizations
        story.append(PageBreak())

        # 4. Dimension Clusters
        fig_clusters = plot_dimension_clusters(embedding)
        fig_clusters.set_dpi(150)
        fig_clusters.tight_layout()
        clusters_img = BytesIO()
        fig_clusters.savefig(clusters_img, format='png', bbox_inches='tight')
        plt.close(fig_clusters)
        clusters_img.seek(0)

        # 5. PCA Visualization
        fig_pca = plot_pca(embedding)
        fig_pca.set_dpi(150)
        fig_pca.tight_layout()
        pca_img = BytesIO()
        fig_pca.savefig(pca_img, format='png', bbox_inches='tight')
        plt.close(fig_pca)
        pca_img.seek(0)

        # Add the remaining visualizations
        add_figure_with_caption(story, clusters_img, "Dimension Clusters Heatmap")
        add_figure_with_caption(story, pca_img, "PCA Visualization of Embedding Segments")

        # Add Claude Analysis section with better formatting
        story.append(PageBreak())
        story.append(Paragraph("Comprehensive Analysis", heading1_style))
        story.append(Spacer(1, 16))

        # Add categorization banner for context
        context_style = ParagraphStyle(
            'ContextStyle',
            parent=normal_style,
            fontName='Helvetica-Bold',
            textColor=colors.white,
            fontSize=10,
            alignment=1,  # Center alignment
            backColor=colors.darkblue
        )

        context_text = f"Analysis tailored for: {business_display} | {page_display}"
        if st.session_state.target_keywords.strip():
            keywords_list = [kw.strip() for kw in st.session_state.target_keywords.split('\n') if kw.strip()]
            context_text += f" | Keywords: {len(keywords_list)} targeted"

        story.append(Paragraph(context_text, context_style))
        story.append(Spacer(1, 12))

        # Process Claude analysis text - convert from HTML/markdown to plain text
        # Remove HTML tags
        analysis_text = re.sub(r'<[^>]*>', '', claude_analysis)

        # Function to process headings better
        def process_heading(text, style):
            text = text.strip()
            if text.startswith('##'):
                return Paragraph(text.replace('##', '').strip(), heading1_style)
            elif text.startswith('#'):
                return Paragraph(text.replace('#', '').strip(), heading2_style)
            elif text.strip() and text.strip()[0].isupper():  # Likely a section title without # marks
                first_line = text.split("\n")[0].strip()
                if len(first_line) < 50 and first_line.isupper():  # It's probably a heading
                    return Paragraph(first_line, heading2_style)
            # Normal paragraph
            return Paragraph(text, normal_style)

        # Split by paragraphs and process each one
        paragraphs = analysis_text.split('\n\n')

        for paragraph in paragraphs:
            if paragraph.strip():
                element = process_heading(paragraph, normal_style)
                story.append(element)
                if isinstance(element, Paragraph) and element.style == heading1_style:
                    story.append(Spacer(1, 12))  # More space after main headings

        # Build the PDF
        doc.build(story)

        # Get the PDF value
        pdf_value = buffer.getvalue()
        buffer.close()

        return pdf_value

    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

# Main content area
def main():
    st.title("SEO Embedding Analysis Tool")

    # Content input options
    st.header("Content Input")

    input_method = st.radio("Choose input method:", ("Paste Content", "Fetch from URL"), key="input_method_radio")

    # Use a placeholder for content that reflects the current input method
    content_placeholder = "Enter the content you want to analyze..." if input_method == "Paste Content" else "Content will be fetched from the URL..."

    if input_method == "Paste Content":
        # Text area for pasting content
        # Value is read directly from session state on rerun
        st.session_state.content = st.text_area("Paste your content here:", height=300,
                              placeholder=content_placeholder,
                              value=st.session_state.get('content', ''), key="pasted_content_area")
        st.session_state.url = "" # Clear URL state when pasting content
        # Button for analyzing pasted content
        analyze_pasted_button = st.button("Analyze Pasted Content")
        analyze_fetch_button = False # Ensure fetch button state doesn't interfere

    else: # Fetch from URL
        # Text input for URL
        url = st.text_input("Enter the URL:", value=st.session_state.get('url', ''),
                           placeholder="e.g., https://www.example.com/your-page")
        # Update session state URL
        st.session_state.url = url
        # Clear pasted content state initially for URL mode on radio button change
        # This is handled implicitly by the input_method check below

        # Button to trigger fetch and then analysis
        analyze_fetch_button = st.button("Fetch and Analyze URL")
        analyze_pasted_button = False # Ensure paste button state doesn't interfere

        # Logic to fetch content when the fetch button is clicked
        if analyze_fetch_button and url:
             with st.spinner(f"Fetching content from {url}..."):
                fetched_content = fetch_content_from_url(url)
                if fetched_content:
                    st.session_state.content = fetched_content # Store fetched content
                    st.success("Content fetched successfully! Running analysis...")
                    st.session_state.fetch_button_clicked = True # Indicate successful fetch for the analysis trigger
                else:
                     st.error("Failed to fetch content. Please check the URL or try pasting.")
                     st.session_state.content = "" # Clear content on failure
                     st.session_state.url = "" # Clear URL on failure
                     st.session_state.fetch_button_clicked = False # Reset state on failure if fetch fails
             # Trigger a rerun after fetch attempt to update the UI and potentially start analysis
             raise RerunException(RerunData()) # Use the correct rerun method with RerunData

        elif analyze_fetch_button and not url:
             st.warning("Please enter a URL.")
             st.session_state.fetch_button_clicked = False # Reset state if button clicked with no URL

    # Display fetched content in a text area for review if available and in URL mode
    # This text area also allows editing the fetched content before analysis if needed
    if st.session_state.content and input_method == "Fetch from URL":
         st.subheader("Fetched Content (Review)")
         # Use a unique key for the text area in URL mode to maintain its state separately
         st.session_state.content = st.text_area("Review and edit fetched content:", value=st.session_state.content, height=300, key="fetched_content_review_area")

    # Enhanced content categorization section with keyword targeting
    st.write("### Content Categorization & SEO Targeting")
    st.write("Categorize your content and specify target keywords for more precise SEO analysis:")

    # Create three columns for the dropdown menus and keyword input
    cat_col1, cat_col2, cat_col3 = st.columns([1, 1, 1.2])

    with cat_col1:
        business_type = st.selectbox(
            "Business Type:",
            options=["Lead Generation/Service", "E-commerce", "SaaS/Tech", "Educational/Informational", "Local Business"],
            index=["Lead Generation/Service", "E-commerce", "SaaS/Tech", "Educational/Informational", "Local Business"].index(
                "Lead Generation/Service" if st.session_state.business_type == "lead_generation" else
                "E-commerce" if st.session_state.business_type == "ecommerce" else
                "SaaS/Tech" if st.session_state.business_type == "saas" else
                "Educational/Informational" if st.session_state.business_type == "educational" else
                "Local Business"
            ),
            help="Select the business model that best matches your content",
            key="business_type_selectbox"
        )

        # Convert display values to internal values
        st.session_state.business_type = (
            "lead_generation" if business_type == "Lead Generation/Service" else
            "ecommerce" if business_type == "E-commerce" else
            "saas" if business_type == "SaaS/Tech" else
            "educational" if business_type == "Educational/Informational" else
            "local_business"
        )

    # Dynamically change page type options based on business type
    with cat_col2:
        if st.session_state.business_type == "lead_generation":
            page_options = ["Landing Page", "Service Page", "Blog Post", "About Us", "Contact Page"]
        elif st.session_state.business_type == "ecommerce":
            page_options = ["Product Page", "Category Page", "Homepage", "Blog Post", "Checkout Page"]
        elif st.session_state.business_type == "saas":
            page_options = ["Feature Page", "Pricing Page", "Homepage", "Blog Post", "Documentation"]
        elif st.session_state.business_type == "educational":
            page_options = ["Course Page", "Resource Page", "Blog Post", "Homepage", "About Us"]
        else:  # local_business
            page_options = ["Homepage", "Service Page", "Location Page", "About Us", "Contact Page"]

        # Get current page type and find its index in the new options, defaulting to 0 if not found
        current_page_display = next((p for p in page_options if p.lower().replace(" ", "_") == st.session_state.page_type), page_options[0])
        current_page_index = page_options.index(current_page_display) if current_page_display in page_options else 0

        page_type = st.selectbox(
            "Page Type:",
            options=page_options,
            index=current_page_index,
            help="Select the type of page you're analyzing",
            key="page_type_selectbox"
        )

        # Convert display values to internal values
        st.session_state.page_type = page_type.lower().replace(" ", "_")

    # New keyword targeting input
    with cat_col3:
        target_keywords = st.text_area(
            "Target Keywords:",
            value=st.session_state.target_keywords,
            height=100,
            placeholder="Enter your target keywords, one per line:\n\ndigital marketing services\nSEO optimization\nbrand strategy\nlocal SEO",
            help="Enter the keywords you want to rank for. Put each keyword on a new line. Include both primary and secondary keywords.",
            key="target_keywords_input"
        )
        
        st.session_state.target_keywords = target_keywords
        
        # Show keyword count and validation
        if target_keywords.strip():
            keywords_list = [kw.strip() for kw in target_keywords.split('\n') if kw.strip()]
            st.caption(f"Keywords entered: {len(keywords_list)}")
            
            # Show a preview of keywords in an expander
            with st.expander("Preview Keywords", expanded=False):
                for i, kw in enumerate(keywords_list[:10], 1):  # Show first 10
                    st.write(f"{i}. {kw}")
                if len(keywords_list) > 10:
                    st.write(f"... and {len(keywords_list) - 10} more")
        else:
            st.caption("Add keywords for targeted analysis")

    # Add keyword strategy tips
    with st.expander("Keyword Strategy Tips", expanded=False):
        st.markdown("""
        **Effective Keyword Targeting:**
        - **Primary Keywords**: 1-3 main keywords you want to rank for
        - **Secondary Keywords**: 5-10 related terms and long-tail variations
        - **Local Keywords**: Include location-based terms for local businesses
        - **Intent-based Keywords**: Match keywords to user search intent
        
        **Examples by Business Type:**
        - **Lead Gen**: "digital marketing agency", "PPC management services"
        - **E-commerce**: "wireless headphones", "bluetooth earbuds under $100"
        - **SaaS**: "project management software", "team collaboration tools"
        - **Educational**: "online courses", "digital marketing certification"
        - **Local**: "dentist near me", "plumber in [city]"
        """)

    # --- Analysis Trigger Logic ---
    # Determine if analysis should be triggered
    # Trigger analysis if either the paste button is clicked and content is available,
    # OR if the fetch button was clicked in the previous run and content is now successfully loaded in session state.
    trigger_analysis = False

    # Case 1: Paste Content and Analyze button clicked
    if input_method == "Paste Content" and analyze_pasted_button and st.session_state.content:
         trigger_analysis = True

    # Case 2: Fetch from URL button clicked and fetch was successful (state updated in previous rerun)
    # We check fetch_button_clicked which was set True in the previous rerun upon successful fetch
    elif input_method == "Fetch from URL" and st.session_state.fetch_button_clicked and st.session_state.content:
         trigger_analysis = True
         # Reset the fetch_button_clicked state immediately after triggering analysis
         st.session_state.fetch_button_clicked = False

    # --- Analysis Execution ---
    # This block runs if the trigger_analysis flag is True based on the conditions above
    if trigger_analysis:

        # Reset PDF data if a new analysis is starting
        st.session_state.pdf_data = None
        st.session_state.pdf_generated = False
        st.session_state.claude_analysis = None # Clear previous analysis results

        # Check if API keys are provided before proceeding with analysis
        if not st.session_state.google_api_key or not st.session_state.anthropic_api_key:
            st.error("Please provide both Google API and Anthropic API keys in the sidebar.")
            # Reset analysis state if keys are missing to prevent displaying old results
            st.session_state.embedding = None
            st.session_state.analysis = None
            st.session_state.claude_analysis = None
            # No rerun needed here, the error message is sufficient to stop
            return

        with st.spinner("Analyzing content with keyword targeting... This may take a minute."):

            # Get embedding with progress indicator
            progress_text = st.empty()
            progress_text.text("Generating embedding from Google Gemini API...")
            st.session_state.embedding = get_embedding(st.session_state.content)

            # Generate analysis
            progress_text.text("Analyzing embedding patterns...")
            st.session_state.analysis = analyze_embedding(st.session_state.embedding)

            # Get Claude analysis with business, page type, and keyword context
            progress_text.text("Getting comprehensive SEO and keyword analysis from Claude...")
            # Increase content sent to Claude to include keyword analysis section
            claude_content_snippet = st.session_state.content[:7000]
            st.session_state.claude_analysis = analyze_with_claude(
                st.session_state.embedding,
                claude_content_snippet,
                st.session_state.business_type,
                st.session_state.page_type
            )
            progress_text.empty()

            # Display results only after successful analysis
            if st.session_state.claude_analysis and "Error getting analysis" not in st.session_state.claude_analysis:
                 st.success("Analysis complete with keyword targeting insights!")
                 # Trigger a rerun to display the results tabs
                 raise RerunException(RerunData())
            else:
                 st.error("Analysis failed. Please check your API keys and try again.")

    # Only display the results if we have a completed analysis in the session state
    # Check for both embedding and claude_analysis to ensure analysis is complete and not an error message
    if st.session_state.embedding is not None and st.session_state.claude_analysis is not None and "Error getting analysis" not in st.session_state.claude_analysis:
        # Create tabs for different sections with keyword insights
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Visualizations", "Metrics", "Keywords", "Clusters", "Analysis Report"])

        # Tab 1: Visualizations
        with tab1:
            st.subheader("Embedding Overview")
            fig1 = plot_embedding_overview(st.session_state.embedding)
            st.pyplot(fig1)
            plt.close(fig1)  # Release memory

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top Dimensions")
                fig2 = plot_top_dimensions(st.session_state.embedding)
                st.pyplot(fig2)
                plt.close(fig2)  # Release memory

            with col2:
                st.subheader("Activation Distribution")
                fig3 = plot_activation_histogram(st.session_state.embedding)
                st.pyplot(fig3)
                plt.close(fig3)  # Release memory

            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Dimension Clusters")
                fig4 = plot_dimension_clusters(st.session_state.embedding)
                st.pyplot(fig4)
                plt.close(fig4)  # Release memory

            with col4:
                st.subheader("PCA Visualization")
                fig5 = plot_pca(st.session_state.embedding)
                st.pyplot(fig5)
                plt.close(fig5)  # Release memory

        # Tab 2: Metrics
        with tab2:
            st.subheader("Key Metrics")
            metrics = st.session_state.analysis["metrics"]

            # Create a 3-column layout for metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Dimensions", metrics["dimension_count"])
                st.metric("Mean Value", f"{metrics['mean_value']:.6f}")
                st.metric("Standard Deviation", f"{metrics['std_dev']:.6f}")

            with col2:
                st.metric("Min Value", f"{metrics['min_value']:.6f} (dim {metrics['min_dimension']})")
                st.metric("Max Value", f"{metrics['max_value']:.6f} (dim {metrics['max_dimension']})")
                st.metric("Median Value", f"{metrics['median_value']:.6f}")

            with col3:
                st.metric("Positive Values", f"{metrics['positive_count']} ({metrics['positive_count']/metrics['dimension_count']*100:.2f}%)")
                st.metric("Negative Values", f"{metrics['negative_count']} ({metrics['negative_count']/metrics['dimension_count']*100:.2f}%)")
                st.metric("Significant Dimensions", f"{metrics['significant_dims']} (>0.1)")

        # Tab 3: NEW Keyword Analysis Tab
        with tab3:
            st.subheader("Keyword Targeting Analysis")
            
            if st.session_state.target_keywords.strip():
                keywords_list = [kw.strip() for kw in st.session_state.target_keywords.split('\n') if kw.strip()]
                
                # Keyword overview
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Keywords", len(keywords_list))
                with col2:
                    st.metric("Primary Keywords", min(3, len(keywords_list)))
                with col3:
                    st.metric("Long-tail Keywords", max(0, len(keywords_list) - 3))
                
                # Display keywords with analysis if content contains keyword analysis
                if "=== KEYWORD TARGETING ANALYSIS ===" in st.session_state.content:
                    st.subheader("Keyword Performance Summary")
                    
                    # Parse keyword analysis from content (this is a simplified approach)
                    # In a production app, you'd want to store this data separately
                    content_lines = st.session_state.content.split('\n')
                    keyword_section_start = False
                    current_keyword = None
                    
                    for line in content_lines:
                        if "=== KEYWORD TARGETING ANALYSIS ===" in line:
                            keyword_section_start = True
                            continue
                        elif "=== CONTENT STATISTICS ===" in line:
                            break
                        elif keyword_section_start and line.startswith("KEYWORD:"):
                            current_keyword = line.replace("KEYWORD:", "").strip()
                            with st.expander(f"Analysis for: {current_keyword}", expanded=False):
                                # Extract keyword data from the following lines
                                # This is a simplified version - you might want to enhance this
                                st.write("**Keyword Analysis:**")
                                st.write("• Detailed analysis available in the full report")
                                st.write("• Check the Analysis Report tab for complete insights")
                else:
                    st.info("Keyword analysis will be included in the next content fetch or analysis.")
                
                # Show keyword list
                st.subheader("Target Keywords List")
                for i, keyword in enumerate(keywords_list, 1):
                    keyword_type = "Primary" if i <= 3 else "Secondary"
                    st.write(f"{i}. **{keyword}** _{keyword_type}_")
                
            else:
                st.info("**No target keywords specified**")
                st.write("Add target keywords in the Content Categorization section to see detailed keyword analysis including:")
                st.write("• Keyword density analysis")
                st.write("• SEO element integration")
                st.write("• Semantic alignment scores") 
                st.write("• Optimization recommendations")
                
                with st.expander("Keyword Research Tips", expanded=False):
                    st.markdown("""
                    **How to choose effective target keywords:**
                    
                    1. **Primary Keywords (1-3)**: Main terms you want to rank for
                    2. **Secondary Keywords (5-10)**: Supporting and long-tail variations
                    3. **Intent Matching**: Choose keywords that match your page purpose
                    4. **Competition Analysis**: Balance search volume with ranking difficulty
                    
                    **Keyword Research Tools:**
                    - Google Keyword Planner
                    - SEMrush / Ahrefs
                    - Google Search Console
                    - Answer The Public
                    """)

        # Tab 4: Clusters
        with tab4:
            st.subheader("Dimension Clusters")

            if not st.session_state.analysis["clusters"]:
                st.info("No significant dimension clusters detected.")
            else:
                for cluster in st.session_state.analysis["clusters"]:
                    with st.expander(f"Cluster #{cluster['id']}: Dimensions {cluster['start_dim']}-{cluster['end_dim']}"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Size", f"{cluster['size']} dimensions")
                        col2.metric("Avg Value", f"{cluster['avg_value']:.6f}")
                        col3.metric("Max Value", f"{cluster['max_value']:.6f} (dim {cluster['max_dim']})")

        # Tab 5: Enhanced Claude Analysis with keyword context
        with tab5:
            # Show content categorization info with keywords
            business_display = {
                "lead_generation": "Lead Generation/Service",
                "ecommerce": "E-commerce",
                "saas": "SaaS/Tech",
                "educational": "Educational/Informational",
                "local_business": "Local Business"
            }.get(st.session_state.business_type, st.session_state.business_type.replace("_", " ").title())

            page_display = st.session_state.page_type.replace("_", " ").title()
            
            # Enhanced context display with keywords
            keywords_display = ""
            if st.session_state.target_keywords.strip():
                keywords_list = [kw.strip() for kw in st.session_state.target_keywords.split('\n') if kw.strip()]
                keywords_display = f"<p style=\"margin: 0;\"><strong>Target Keywords:</strong> {', '.join(keywords_list[:5])}"
                if len(keywords_list) > 5:
                    keywords_display += f" (+{len(keywords_list) - 5} more)"
                keywords_display += "</p>"
            else:
                keywords_display = "<p style=\"margin: 0;\"><strong>Target Keywords:</strong> <em>None specified</em></p>"

            st.markdown(f"""
            <div style="background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <p style="margin: 0; font-weight: bold; color: #1f4e79;">Analysis Context:</p>
                <p style="margin: 0;"><strong>Business Type:</strong> {business_display}</p>
                <p style="margin: 0;"><strong>Page Type:</strong> {page_display}</p>
                {keywords_display}
            </div>
            """, unsafe_allow_html=True)

            st.subheader("Comprehensive SEO & Keyword Analysis Report")

            # Display the analysis content
            st.markdown(st.session_state.claude_analysis, unsafe_allow_html=True)

            # Enhanced download options
            st.write("---")

            with st.container():
                st.markdown("""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
                <h3 style="margin-top: 0;">Download Options</h3>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                # Text download button
                with col1:
                    st.download_button(
                        label="Download as Text",
                        data=st.session_state.claude_analysis,
                        file_name="seo_keyword_analysis_report.txt",
                        mime="text/plain",
                        help="Download the analysis as a plain text file",
                        key="download_text_button"
                    )

                # PDF generation button and download
                with col2:
                    if st.session_state.pdf_data is None:
                        if st.button("Generate PDF Report", help="Create a PDF with visualizations and analysis", key="generate_pdf_button_tab"):
                            with st.spinner("Generating comprehensive PDF report with keyword analysis... This may take up to a minute."):
                                try:
                                    pdf_bytes = create_report_pdf(
                                        st.session_state.embedding,
                                        st.session_state.analysis,
                                        st.session_state.claude_analysis,
                                        st.session_state.business_type,
                                        st.session_state.page_type
                                    )
                                    st.session_state.pdf_data = pdf_bytes
                                    st.session_state.pdf_generated = True
                                    st.success("PDF report with keyword insights generated successfully!")
                                    # Trigger a rerun to show the download button immediately
                                    raise RerunException(RerunData())
                                except Exception as e:
                                    st.error(f"Error generating PDF: {str(e)}")
                    else:
                         # Show PDF download button if PDF has been generated
                         b64_pdf = base64.b64encode(st.session_state.pdf_data).decode('utf-8')
                         download_link = f'<a href="data:application/pdf;base64,{b64_pdf}" download="seo_keyword_analysis_report.pdf" class="button" style="display: inline-block; padding: 12px 20px; background-color: #0c6b58; color: white; text-decoration: none; font-weight: bold; border-radius: 4px; text-align: center; margin: 10px 0; width: 100%;">DOWNLOAD COMPLETE PDF REPORT</a>'
                         st.markdown(download_link, unsafe_allow_html=True)

                         if st.button("Generate New PDF", key="discard_pdf_button_tab"):
                            st.session_state.pdf_data = None
                            st.session_state.pdf_generated = False
                            # Trigger a rerun to update the button state
                            raise RerunException(RerunData())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        # Add a basic content area to ensure something is visible
        st.write("## SEO Embedding Analysis Tool")
        st.write("This tool analyzes content semantics using embeddings.")
