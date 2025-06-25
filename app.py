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

import streamlit.runtime.scriptrunner as rst
from streamlit.runtime.scriptrunner import RerunData, RerunException

st.set_page_config(
    page_title="SEO Embedding Analysis Tool",
    layout="wide"
)

# --- SESSION STATE INIT ---
try:
    if 'google_api_key' not in st.session_state:
        st.session_state.google_api_key = st.secrets.get("GOOGLE_API_KEY", "")
    if 'anthropic_api_key' not in st.session_state:
        st.session_state.anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
except Exception:
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
# ---- TARGET KEYWORD SESSION STATE (NEW) ----
if 'target_keyword' not in st.session_state:
    st.session_state.target_keyword = ""

# ... [Sidebar, NumpyEncoder, get_embedding, get_current_settings, plotting, PDF, and analyze_embedding functions unchanged] ...

# --- ENHANCED METADATA EXTRACTION ---
def fetch_content_from_url(url):
    """Fetches and parses content from a given URL with enhanced metadata extraction."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        response = requests.get(url, headers=headers, timeout=15)
        st.write(f"Debug Info: Status Code: {response.status_code}")
        st.write(f"Debug Info: Response Length: {len(response.content)} bytes")
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # SEO Metadata extraction
        title = soup.find('title')
        title_text = title.get_text(strip=True) if title else "No title found"
        description = soup.find('meta', attrs={'name': 'description'})
        description_text = description.get('content', '').strip() if description else "No meta description found"
        h1_tags = soup.find_all('h1')
        h1_texts = [h1.get_text(strip=True) for h1 in h1_tags if h1.get_text(strip=True)]
        h2_tags = soup.find_all('h2')
        h2_texts = [h2.get_text(strip=True) for h2 in h2_tags if h2.get_text(strip=True)]
        h3_tags = soup.find_all('h3')
        h3_texts = [h3.get_text(strip=True) for h3 in h3_tags if h3.get_text(strip=True)]

        # Content extraction strategies
        main_content = soup.find('main') or soup.find('article')
        if not main_content:
            main_content = soup.find('div', class_=lambda x: x and any(
                keyword in x.lower() for keyword in ['content', 'main', 'article', 'post', 'entry']
            ))
        if not main_content:
            main_content = soup.find('div', id=lambda x: x and any(
                keyword in x.lower() for keyword in ['content', 'main', 'article', 'post']
            ))
        if not main_content and soup.body:
            main_content = soup.body

        text = (main_content.get_text(separator='\n', strip=True) if main_content
                else soup.get_text(separator='\n', strip=True))

        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()

        if len(text) < 100:
            st.warning(f"Warning: Only extracted {len(text)} characters. The page might be JavaScript-heavy or protected.")

        extracted_info = ""
        extracted_info += f"Title: {title_text}\n\n"
        extracted_info += f"Meta Description: {description_text}\n\n"

        if h1_texts:
            extracted_info += f"H1 Tags ({len(h1_texts)} found):\n"
            for i, h1 in enumerate(h1_texts, 1):
                extracted_info += f"  {i}. {h1}\n"
            extracted_info += "\n"
        if h2_texts:
            extracted_info += f"H2 Tags ({len(h2_texts)} found):\n"
            for i, h2 in enumerate(h2_texts[:5], 1):
                extracted_info += f"  {i}. {h2}\n"
            if len(h2_texts) > 5:
                extracted_info += f"  ... and {len(h2_texts) - 5} more H2 tags\n"
            extracted_info += "\n"
        if h3_texts:
            extracted_info += f"H3 Tags ({len(h3_texts)} found):\n"
            for i, h3 in enumerate(h3_texts[:3], 1):
                extracted_info += f"  {i}. {h3}\n"
            if len(h3_texts) > 3:
                extracted_info += f"  ... and {len(h3_texts) - 3} more H3 tags\n"
            extracted_info += "\n"

        extracted_info += "=== PAGE CONTENT ===\n\n"
        extracted_info += text

        st.success("SEO Metadata extracted successfully!")
        with st.expander("View Extracted SEO Metadata", expanded=True):
            st.write(f"**Title:** {title_text}")
            st.write(f"**Meta Description:** {description_text}")
            st.write(f"**H1 Tags:** {len(h1_texts)} found")
            if h1_texts:
                for i, h1 in enumerate(h1_texts, 1):
                    st.write(f"  {i}. {h1}")
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

# --- ANALYZE WITH CLAUDE, ENHANCED WITH KEYWORD ---
def analyze_with_claude(embedding_data, content_snippet, business_type, page_type, target_keyword):
    """Get analysis from Claude with business, page type, and keyword targeting context"""
    try:
        current_settings = get_current_settings()
        anthropic_client = Anthropic(api_key=st.session_state.anthropic_api_key)

        # Business type context
        business_context = ""
        if business_type == "lead_generation":
            business_context = "a lead generation or service-based business focused on converting visitors to leads or clients"
        elif business_type == "ecommerce":
            business_context = "an e-commerce business focused on selling products online and maximizing conversions"
        elif business_type == "saas":
            business_context = "a SaaS or technology company focused on showcasing features and driving sign-ups"
        elif business_type == "educational":
            business_context = "an educational platform or information resource focused on providing valuable content"
        else:
            business_context = "a local business focused on driving local customers and in-person visits"

        # Page type context
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

        # Keyword context
        keyword_context = ""
        if target_keyword and target_keyword.strip():
            keyword_context = f"""

TARGET KEYWORD ANALYSIS:
The target keyword for this content is: "{target_keyword.strip()}"

Please provide specific analysis on:
1. **Keyword Optimization**: How well the content is optimized for this target keyword
2. **Keyword Density**: Analysis of keyword frequency and natural integration
3. **Semantic Relevance**: How semantically related the content is to the target keyword
4. **Title Tag Optimization**: Whether the target keyword appears in the title tag effectively
5. **Meta Description Optimization**: How well the meta description incorporates the target keyword
6. **Header Tag Strategy**: Analysis of keyword usage in H1, H2, H3 tags
7. **Content Gap Analysis**: Missing semantic terms and related keywords that should be included
8. **Keyword Variations**: Suggestions for long-tail variations and semantic keywords
9. **Search Intent Matching**: Whether the content matches the search intent for this keyword
10. **Competitive Positioning**: How to better position content for this keyword"""

        system_prompt = f"""You are an advanced SEO and NLP Embedding Analysis Expert with deep expertise in semantic content optimization, machine learning-driven content strategy, and advanced natural language processing techniques.

Your mission is to provide a comprehensive, multi-dimensional analysis of embedding data that transforms raw numerical information into actionable SEO and content strategy insights.

IMPORTANT CONTEXT: The content being analyzed is for {business_context}. Specifically, it is {page_context}.{keyword_context}
[...all other methodology and structure instructions as before, unchanged...]
"""

        message = anthropic_client.messages.create(
            model=current_settings["model"],
            max_tokens=current_settings["max_tokens"],
            temperature=current_settings["temperature"],
            thinking={
                "type": "enabled",
                "budget_tokens": current_settings["thinking_tokens"]
            },
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"""Comprehensive Embedding Analysis Request:

CONTENT CONTEXT:
- First 4500 characters of content:
{content_snippet[:4500]}

TARGET KEYWORD: {target_keyword if target_keyword and target_keyword.strip() else "Not specified"}

EMBEDDING DATA DIAGNOSTICS:
- Total Dimensions: {len(embedding_data)}
- Dimensional Statistics:
  * Mean Value: {np.mean(embedding_data):.6f}
  * Standard Deviation: {np.std(embedding_data):.6f}
  * Minimum Value: {np.min(embedding_data):.6f} (Dimension {np.argmin(embedding_data)})
  * Maximum Value: {np.max(embedding_data):.6f} (Dimension {np.argmax(embedding_data)})

TOP ACTIVATION DIMENSIONS:
{sorted(range(len(embedding_data)), key=lambda i: abs(embedding_data[i]), reverse=True)[:10]}

BUSINESS TYPE: {business_type}
PAGE TYPE: {page_type}

Please provide a comprehensive analysis following the exact format in your instructions:
1. First, deliver a detailed EMBEDDING ANALYSIS with clear sections and quantitative metrics
2. Then, provide completely separate ACTIONABLE RECOMMENDATIONS with implementation examples specifically tailored for this business type and page type
3. Finally, include a PLAIN LANGUAGE SUMMARY in simple bullet points

Ensure your analysis transforms embedding data into strategic, implementable content optimization insights."""
                }
            ]
        )
        # [Return content as before]
        if hasattr(message.content, '__iter__') and not isinstance(message.content, str):
            extracted_text = ""
            for block in message.content:
                if hasattr(block, 'text'):
                    extracted_text += block.text
                elif isinstance(block, str):
                    extracted_text += block
            return extracted_text
        elif hasattr(message.content, 'text'):
            return message.content.text
        else:
            return str(message.content)
    except Exception as e:
        st.error(f"Error getting Claude analysis: {e}")
        return "Error getting analysis from Claude. Please check your API key and try again."

# --- MAIN ---
def main():
    st.title("SEO Embedding Analysis Tool")
    st.header("Content Input")

    input_method = st.radio("Choose input method:", ("Paste Content", "Fetch from URL"), key="input_method_radio")
    content_placeholder = "Enter the content you want to analyze..." if input_method == "Paste Content" else "Content will be fetched from the URL..."

    if input_method == "Paste Content":
        st.session_state.content = st.text_area("Paste your content here:", height=300,
                              placeholder=content_placeholder,
                              value=st.session_state.get('content', ''), key="pasted_content_area")
        st.session_state.url = ""
        analyze_pasted_button = st.button("Analyze Pasted Content")
        analyze_fetch_button = False
    else:
        url = st.text_input("Enter the URL:", value=st.session_state.get('url', ''),
                           placeholder="e.g., https://www.example.com/your-page")
        st.session_state.url = url
        analyze_fetch_button = st.button("Fetch and Analyze URL")
        analyze_pasted_button = False
        if analyze_fetch_button and url:
             with st.spinner(f"Fetching content from {url}..."):
                fetched_content = fetch_content_from_url(url)
                if fetched_content:
                    st.session_state.content = fetched_content
                    st.success("Content fetched successfully! Running analysis...")
                    st.session_state.fetch_button_clicked = True
                else:
                     st.error("Failed to fetch content. Please check the URL or try pasting.")
                     st.session_state.content = ""
                     st.session_state.url = ""
                     st.session_state.fetch_button_clicked = False
             raise RerunException(RerunData())
        elif analyze_fetch_button and not url:
             st.warning("Please enter a URL.")
             st.session_state.fetch_button_clicked = False

    trigger_analysis = False
    if input_method == "Paste Content" and analyze_pasted_button and st.session_state.content:
         trigger_analysis = True
    elif input_method == "Fetch from URL" and st.session_state.fetch_button_clicked and st.session_state.content:
         trigger_analysis = True
         st.session_state.fetch_button_clicked = False

    if st.session_state.content and input_method == "Fetch from URL":
         st.subheader("Fetched Content (Review)")
         st.session_state.content = st.text_area("Review and edit fetched content:", value=st.session_state.content, height=300, key="fetched_content_review_area")

    # --- ENHANCED CATEGORIZATION & KEYWORD INPUT ---
    st.write("### Content Categorization & SEO Targeting")
    st.write("Categorize your content and specify your target keyword to receive more tailored recommendations:")
    cat_col1, cat_col2, cat_col3 = st.columns(3)

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
        st.session_state.business_type = (
            "lead_generation" if business_type == "Lead Generation/Service" else
            "ecommerce" if business_type == "E-commerce" else
            "saas" if business_type == "SaaS/Tech" else
            "educational" if business_type == "Educational/Informational" else
            "local_business"
        )

    with cat_col2:
        if st.session_state.business_type == "lead_generation":
            page_options = ["Landing Page", "Service Page", "Blog Post", "About Us", "Contact Page"]
        elif st.session_state.business_type == "ecommerce":
            page_options = ["Product Page", "Category Page", "Homepage", "Blog Post", "Checkout Page"]
        elif st.session_state.business_type == "saas":
            page_options = ["Feature Page", "Pricing Page", "Homepage", "Blog Post", "Documentation"]
        elif st.session_state.business_type == "educational":
            page_options = ["Course Page", "Resource Page", "Blog Post", "Homepage", "About Us"]
        else:
            page_options = ["Homepage", "Service Page", "Location Page", "About Us", "Contact Page"]
        current_page_display = next((p for p in page_options if p.lower().replace(" ", "_") == st.session_state.page_type), page_options[0])
        current_page_index = page_options.index(current_page_display) if current_page_display in page_options else 0
        page_type = st.selectbox(
            "Page Type:",
            options=page_options,
            index=current_page_index,
            help="Select the type of page you're analyzing",
            key="page_type_selectbox"
        )
        st.session_state.page_type = page_type.lower().replace(" ", "_")

    # --- KEYWORD INPUT ---
    with cat_col3:
        target_keyword = st.text_input(
            "Target Keyword:",
            value=st.session_state.get('target_keyword', ''),
            placeholder="e.g., best running shoes, digital marketing services",
            help="Enter the main keyword you want this page to rank for. This will help tailor the SEO analysis and recommendations.",
            key="target_keyword_input"
        )
        st.session_state.target_keyword = target_keyword
    if st.session_state.target_keyword and st.session_state.target_keyword.strip():
        st.info(f"Target Keyword: {st.session_state.target_keyword.strip()}")

    # --- ANALYSIS EXECUTION ---
    if trigger_analysis:
        st.session_state.pdf_data = None
        st.session_state.pdf_generated = False
        st.session_state.claude_analysis = None
        if not st.session_state.google_api_key or not st.session_state.anthropic_api_key:
            st.error("Please provide both Google API and Anthropic API keys in the sidebar.")
            st.session_state.embedding = None
            st.session_state.analysis = None
            st.session_state.claude_analysis = None
            return
        with st.spinner("Analyzing content... This may take a minute."):
            progress_text = st.empty()
            progress_text.text("Generating embedding from Google Gemini API...")
            st.session_state.embedding = get_embedding(st.session_state.content)
            progress_text.text("Analyzing embedding patterns...")
            st.session_state.analysis = analyze_embedding(st.session_state.embedding)
            progress_text.text("Getting comprehensive analysis from Claude...")
            claude_content_snippet = st.session_state.content[:4500]
            st.session_state.claude_analysis = analyze_with_claude(
                st.session_state.embedding,
                claude_content_snippet,
                st.session_state.business_type,
                st.session_state.page_type,
                st.session_state.target_keyword
            )
            progress_text.empty()
            if st.session_state.claude_analysis and "Error getting analysis" not in st.session_state.claude_analysis:
                 st.success("Analysis complete!")
                 raise RerunException(RerunData())
            else:
                 st.error("Analysis failed. Please check your API keys and try again.")

    if st.session_state.embedding is not None and st.session_state.claude_analysis is not None and "Error getting analysis" not in st.session_state.claude_analysis:
        tab1, tab2, tab3, tab4 = st.tabs(["Visualizations", "Metrics", "Clusters", "Analysis Report"])
        # [Visualization, metrics, clusters unchanged...]
        with tab4:
            business_display = {
                "lead_generation": "Lead Generation/Service",
                "ecommerce": "E-commerce",
                "saas": "SaaS/Tech",
                "educational": "Educational/Informational",
                "local_business": "Local Business"
            }.get(st.session_state.business_type, st.session_state.business_type.replace("_", " ").title())
            page_display = st.session_state.page_type.replace("_", " ").title()
            # --- CONTEXT DISPLAY (ENHANCED) ---
            context_html = f"""
            Analysis tailored for:
            Business Type: {business_display}
            Page Type: {page_display}"""
            if st.session_state.target_keyword and st.session_state.target_keyword.strip():
                context_html += f"""
            Target Keyword: {st.session_state.target_keyword.strip()}"""
            context_html += """
            """
            st.markdown(context_html, unsafe_allow_html=True)
            st.subheader("Comprehensive Embedding Analysis Report")
            st.markdown(st.session_state.claude_analysis, unsafe_allow_html=True)
            st.write("---")
            with st.container():
                st.markdown("""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
                <h3 style="margin-top: 0;">Download Options</h3>
                </div>
                """, unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download as Text",
                        data=st.session_state.claude_analysis,
                        file_name="seo_embedding_analysis_report.txt",
                        mime="text/plain",
                        help="Download the analysis as a plain text file",
                        key="download_text_button"
                    )
                with col2:
                    if st.session_state.pdf_data is None:
                        if st.button("Generate PDF Report", help="Create a PDF with visualizations and analysis", key="generate_pdf_button_tab"):
                            with st.spinner("Generating PDF report... This may take up to a minute."):
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
                                    st.success("PDF report generated successfully!")
                                    raise RerunException(RerunData())
                                except Exception as e:
                                    st.error(f"Error generating PDF: {str(e)}")
                    else:
                         b64_pdf = base64.b64encode(st.session_state.pdf_data).decode('utf-8')
                         download_link = f'<a href="data:application/pdf;base64,{b64_pdf}" download="seo_embedding_analysis_report.pdf" class="button" style="display: inline-block; padding: 12px 20px; background-color: #0c6b58; color: white; text-decoration: none; font-weight: bold; border-radius: 4px; text-align: center; margin: 10px 0; width: 100%;">DOWNLOAD COMPLETE PDF REPORT</a>'
                         st.markdown(download_link, unsafe_allow_html=True)
                         if st.button("Discard PDF and Generate Again", key="discard_pdf_button_tab"):
                            st.session_state.pdf_data = None
                            st.session_state.pdf_generated = False
                            raise RerunException(RerunData())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("## SEO Embedding Analysis Tool")
        st.write("This tool analyzes content semantics using embeddings.")
