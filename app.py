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
from datetime import datetime

# Import the necessary module and class for programmatic rerun
import streamlit.runtime.scriptrunner as rst
from streamlit.runtime.scriptrunner import RerunData, RerunException

# Configure page
st.set_page_config(
    page_title="SEO Embedding Analysis Tool",
    layout="wide"
)

# Initialize session state for settings if they don't exist
try:
    if 'google_api_key' not in st.session_state:
        st.session_state.google_api_key = st.secrets.get("GOOGLE_API_KEY", "") [cite: 2]
    if 'anthropic_api_key' not in st.session_state:
        st.session_state.anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", "") [cite: 2]
except Exception as e:
    if 'google_api_key' not in st.session_state:
        st.session_state.google_api_key = "" [cite: 2]
    if 'anthropic_api_key' not in st.session_state:
        st.session_state.anthropic_api_key = "" [cite: 3]

if 'claude_model' not in st.session_state:
    st.session_state.claude_model = "claude-3-7-sonnet-latest"
if 'max_tokens' not in st.session_state:
    st.session_state.max_tokens = 15000 [cite: 3]
if 'temperature' not in st.session_state:
    st.session_state.temperature = 1.0 [cite: 3]
if 'thinking_tokens' not in st.session_state:
    st.session_state.thinking_tokens = 8000 [cite: 3]
if 'embedding' not in st.session_state:
    st.session_state.embedding = None [cite: 3]
if 'analysis' not in st.session_state:
    st.session_state.analysis = None [cite: 3]
if 'claude_analysis' not in st.session_state:
    st.session_state.claude_analysis = None [cite: 3]
if 'content' not in st.session_state:
    st.session_state.content = "" [cite: 3]
if 'pdf_data' not in st.session_state:
    st.session_state.pdf_data = None [cite: 4]
if 'pdf_generated' not in st.session_state:
    st.session_state.pdf_generated = False [cite: 4]
if 'business_type' not in st.session_state:
    st.session_state.business_type = "lead_generation" [cite: 4]
if 'page_type' not in st.session_state:
    st.session_state.page_type = "landing_page" [cite: 4]
if 'url' not in st.session_state:
    st.session_state.url = "" [cite: 4]
# NEW: Initialize target_keyword in session state
if 'target_keyword' not in st.session_state:
    st.session_state.target_keyword = ""
if 'fetch_button_clicked' not in st.session_state:
    st.session_state.fetch_button_clicked = False [cite: 4]

# Sidebar for API keys and settings
with st.sidebar:
    st.title("API Settings")
    google_api_key = st.text_input("Google API Key", type="password", value=st.session_state.google_api_key) [cite: 4, 5]
    anthropic_api_key = st.text_input("Anthropic API Key", type="password", value=st.session_state.anthropic_api_key) [cite: 5]

    st.subheader("Model Settings")
    claude_model = st.selectbox("Claude Model", ["claude-3-7-sonnet-latest", "claude-3-opus-20240229", "claude-3-5-sonnet-20240620"], index=0 if st.session_state.claude_model == "claude-3-7-sonnet-latest" else 1 if st.session_state.claude_model == "claude-3-opus-20240229" else 2) [cite: 5, 6]
    max_tokens = st.slider("Max Tokens", 4000, 15000, st.session_state.max_tokens, 1000) [cite: 6]
    temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.1) [cite: 6]
    thinking_tokens = st.slider("Thinking Tokens", 3000, 8000, st.session_state.thinking_tokens, 1000) [cite: 6]

    if st.button("Save Settings", type="primary"):
        st.session_state.google_api_key = google_api_key [cite: 6]
        st.session_state.anthropic_api_key = anthropic_api_key [cite: 7]
        st.session_state.claude_model = claude_model [cite: 7]
        st.session_state.max_tokens = max_tokens [cite: 7]
        st.session_state.temperature = temperature [cite: 7]
        st.session_state.thinking_tokens = thinking_tokens [cite: 7]
        st.success("Settings saved!")

# Custom JSON encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj) [cite: 7]
        if isinstance(obj, np.floating):
            return float(obj) [cite: 8]
        if isinstance(obj, np.ndarray):
            return obj.tolist() [cite: 8]
        return super(NumpyEncoder, self).default(obj) [cite: 8]

def get_embedding(text):
    try:
        genai.configure(api_key=st.session_state.google_api_key) [cite: 8]
        response = genai.embed_content(model="models/gemini-embedding-exp-03-07", content=text) [cite: 9]
        return response["embedding"] [cite: 9]
    except Exception as e:
        st.error(f"Error getting embedding: {e}")
        st.warning("Using random embedding instead")
        return np.random.normal(0, 0.1, 3072).tolist() [cite: 10]

def get_current_settings():
    return {
        "model": st.session_state.claude_model,
        "max_tokens": st.session_state.max_tokens,
        "temperature": st.session_state.temperature,
        "thinking_tokens": st.session_state.thinking_tokens
    } [cite: 10]

# ENHANCED: Function updated to accept and use target_keyword
def analyze_with_claude(embedding_data, content_snippet, business_type, page_type, target_keyword):
    """Get analysis from Claude with business, page type, and keyword context."""
    try:
        current_settings = get_current_settings() [cite: 11]
        anthropic_client = Anthropic(api_key=st.session_state.anthropic_api_key) [cite: 11]

        business_context = {
            "lead_generation": "a lead generation or service-based business focused on converting visitors to leads or clients",
            "ecommerce": "an e-commerce business focused on selling products online and maximizing conversions",
            "saas": "a SaaS or technology company focused on showcasing features and driving sign-ups",
            "educational": "an educational platform or information resource focused on providing valuable content",
            "local_business": "a local business focused on driving local customers and in-person visits"
        }.get(business_type, "a general business") [cite: 11, 12, 13]

        page_context = f"a {page_type.replace('_', ' ')} designed for a specific purpose within the website" [cite: 13, 14, 15, 16, 17]

        # NEW: Create keyword context to guide the analysis
        keyword_context = f"The primary target keyword for this page is '{target_keyword}'. All analysis should be performed with this keyword in mind." if target_keyword else "No specific target keyword has been provided; analyze for the dominant semantic themes."

        message = anthropic_client.messages.create(
            model=current_settings["model"],
            max_tokens=current_settings["max_tokens"],
            temperature=current_settings["temperature"],
            system=f"""You are an advanced SEO and NLP Embedding Analysis Expert.
IMPORTANT CONTEXT: The content being analyzed is for {business_context}. Specifically, it is {page_context}.
{keyword_context}
Your mission is to provide a comprehensive, multi-dimensional analysis of embedding data and page content to generate actionable SEO insights.
Tailor all your analysis and recommendations to the specific business, page type, and TARGET KEYWORD provided.
Your output must follow this structure precisely:
1. **Contextual Explanation**: Explain embedding dimensions and metrics.
2. **Embedding Analysis**: Technical analysis of patterns.
3. **Actionable Recommendations**: Specific, implementable suggestions tailored to the business, page type, and keyword.
4. **Summary**: Key findings in non-technical language.""", [cite: 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
            messages=[
                {
                    "role": "user",
                    "content": f"""Comprehensive Embedding Analysis Request:

CONTENT & METADATA:
(Content may include SEO metadata like Title, H1s, etc., under '=== SEO METADATA ===' heading)
{content_snippet[:4500]}

EMBEDDING DATA DIAGNOSTICS:
- Total Dimensions: {len(embedding_data)}
- Mean Value: {np.mean(embedding_data):.6f}
- Standard Deviation: {np.std(embedding_data):.6f}

CONTEXT FOR ANALYSIS:
- BUSINESS TYPE: {business_type}
- PAGE TYPE: {page_type}
- TARGET KEYWORD: {target_keyword if target_keyword else 'Not Provided'}

Please provide a comprehensive analysis following the exact format in your instructions. Focus heavily on how well the content and its embedding align with the target keyword '{target_keyword}'.
"""
                }
            ]
        )

        if hasattr(message.content, '__iter__') and not isinstance(message.content, str):
            return "".join(block.text for block in message.content if hasattr(block, 'text')) [cite: 31, 32]
        else:
            return str(message.content) [cite: 33]

    except Exception as e:
        st.error(f"Error getting Claude analysis: {e}")
        return "Error getting analysis from Claude. Please check your API key and try again." [cite: 34]

# ENHANCED: New fetch_content_from_url function with detailed metadata extraction
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
        } [cite: 87, 88]
        
        response = requests.get(url, headers=headers, timeout=15) [cite: 89]
        st.write(f"Debug Info: Status Code: {response.status_code}") [cite: 89]
        st.write(f"Debug Info: Response Length: {len(response.content)} bytes") [cite: 89]
        response.raise_for_status() [cite: 89]
        
        soup = BeautifulSoup(response.content, 'html.parser') [cite: 90]
        
        for script in soup(["script", "style"]):
            script.decompose() [cite: 90]
        
        # Extract SEO metadata
        title = soup.find('title') [cite: 96]
        title_text = title.get_text(strip=True) if title else "No title found"
        
        description = soup.find('meta', attrs={'name': 'description'}) [cite: 96]
        description_text = description.get('content', '').strip() if description else "No meta description found"
        
        h1_tags = [h1.get_text(strip=True) for h1 in soup.find_all('h1') if h1.get_text(strip=True)]
        h2_tags = [h2.get_text(strip=True) for h2 in soup.find_all('h2') if h2.get_text(strip=True)]
        h3_tags = [h3.get_text(strip=True) for h3 in soup.find_all('h3') if h3.get_text(strip=True)]
        
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=lambda x: x and any(k in x.lower() for k in ['content', 'main', 'article', 'post', 'entry'])) [cite: 91, 92]
        if not main_content and soup.body:
            main_content = soup.body [cite: 93]
        
        text = main_content.get_text(separator='\n', strip=True) if main_content else soup.get_text(separator='\n', strip=True) [cite: 94]
        
        text = re.sub(r'\n\s*\n', '\n\n', text) [cite: 94]
        text = re.sub(r'[ \t]+', ' ', text).strip() [cite: 95]
        
        # Build comprehensive extracted information with SEO metadata prominently featured
        extracted_info = "=== SEO METADATA ===\n\n"
        extracted_info += f"Title Tag: {title_text}\n\n"
        extracted_info += f"Meta Description: {description_text}\n\n"
        
        if h1_tags:
            extracted_info += f"H1 Tags ({len(h1_tags)} found):\n" + "\n".join(f"  - {h1}" for h1 in h1_tags) + "\n\n"
        else:
            extracted_info += "H1 Tags: No H1 tags found\n\n"

        if h2_tags:
            extracted_info += f"H2 Tags ({len(h2_tags)} found):\n" + "\n".join(f"  - {h2}" for h2 in h2_tags[:5]) + "\n\n"

        if h3_tags:
             extracted_info += f"H3 Tags ({len(h3_tags)} found):\n" + "\n".join(f"  - {h3}" for h3 in h3_tags[:3]) + "\n\n"

        extracted_info += "=== PAGE CONTENT ===\n\n"
        extracted_info += text
        
        st.success("SEO Metadata extracted successfully!")
        with st.expander("View Extracted SEO Metadata", expanded=True):
            st.write(f"**Title:** {title_text}")
            st.write(f"**Meta Description:** {description_text}")
            if h1_tags:
                st.write(f"**H1 Tags:**")
                st.markdown('\n'.join(f"- {h1}" for h1 in h1_tags))
        
        return extracted_info
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching or parsing URL: {e}") [cite: 98, 99]
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during content fetching: {e}") [cite: 99]
        return None

# PLOTTING AND ANALYSIS FUNCTIONS (Mostly unchanged)
def plot_embedding_overview(embedding):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(embedding)), embedding)
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax.set_title('Embedding Values Across All 3k Dimensions') [cite: 34]
    ax.set_xlabel('Dimension') [cite: 35]
    ax.set_ylabel('Value') [cite: 35]
    ax.grid(True, alpha=0.3)
    return fig

def plot_top_dimensions(embedding):
    top_indices = sorted(range(len(embedding)), key=lambda i: abs(embedding[i]), reverse=True)[:20] [cite: 36]
    top_values = [embedding[i] for i in top_indices] [cite: 36]
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['blue' if v >= 0 else 'red' for v in top_values] [cite: 36]
    ax.bar(range(len(top_indices)), top_values, color=colors) [cite: 36]
    ax.set_xticks(range(len(top_indices))) [cite: 37]
    ax.set_xticklabels(top_indices, rotation=45) [cite: 37]
    ax.set_title('Top 20 Dimensions by Magnitude') [cite: 37]
    ax.set_xlabel('Dimension Index') [cite: 37]
    ax.set_ylabel('Value') [cite: 37]
    ax.grid(True, alpha=0.3)
    return fig

def plot_dimension_clusters(embedding):
    embedding_reshaped = np.array(embedding).reshape(64, 48) [cite: 38]
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = LinearSegmentedColormap.from_list('BrBG', ['blue', 'white', 'red'], N=256) [cite: 39]
    im = ax.imshow(embedding_reshaped, cmap=cmap, aspect='auto') [cite: 39]
    plt.colorbar(im, ax=ax, label='Activation Value') [cite: 39]
    ax.set_title('Embedding Clusters Heatmap (Reshaped to 64x48)') [cite: 39]
    ax.set_xlabel('Dimension Group') [cite: 39]
    ax.set_ylabel('Dimension Group') [cite: 39]
    return fig

def plot_pca(embedding):
    segment_size = 256
    num_segments = len(embedding) // segment_size [cite: 41]
    data_matrix = np.array([embedding[i*segment_size:(i+1)*segment_size] for i in range(num_segments)]) [cite: 41]
    fig, ax = plt.subplots(figsize=(10, 8))
    if num_segments > 1:
        pca = PCA(n_components=2) [cite: 42]
        pca_results = pca.fit_transform(data_matrix) [cite: 42]
        ax.scatter(pca_results[:, 0], pca_results[:, 1]) [cite: 42]
        for i in range(num_segments):
            start = i * segment_size [cite: 43]
            end = start + segment_size - 1 [cite: 43]
            ax.annotate(f"{start}-{end}", (pca_results[i, 0], pca_results[i, 1]), fontsize=8) [cite: 43, 44]
        ax.set_title('PCA of Embedding Segments') [cite: 44]
        ax.set_xlabel('Principal Component 1') [cite: 44]
        ax.set_ylabel('Principal Component 2') [cite: 44]
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Not enough segments for PCA visualization", ha='center', va='center', fontsize=12) [cite: 45]
        ax.axis('off')
    return fig

def plot_activation_histogram(embedding):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(embedding, bins=50, alpha=0.7, color='skyblue', edgecolor='black') [cite: 46]
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.7) [cite: 46]
    ax.set_title('Distribution of Embedding Values') [cite: 46]
    ax.set_xlabel('Value') [cite: 47]
    ax.set_ylabel('Frequency') [cite: 47]
    ax.grid(True, alpha=0.3)
    return fig

def analyze_embedding(embedding):
    embedding = np.array(embedding) [cite: 48]
    abs_embedding = np.abs(embedding) [cite: 48]
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
    } [cite: 48, 49, 50]
    
    significant_dims = np.where(abs_embedding > 0.1)[0] [cite: 50]
    clusters = []
    if len(significant_dims) > 0:
        current_cluster = [int(significant_dims[0])] [cite: 51]
        for i in range(1, len(significant_dims)):
            if significant_dims[i] - significant_dims[i-1] <= 5:
                current_cluster.append(int(significant_dims[i])) [cite: 52]
            else:
                if len(current_cluster) > 1: clusters.append(current_cluster)
                current_cluster = [int(significant_dims[i])] [cite: 52, 53]
        if len(current_cluster) > 1: clusters.append(current_cluster)
    
    cluster_info = [{
        "id": i+1,
        "dimensions": c,
        "start_dim": int(min(c)),
        "end_dim": int(max(c)),
        "size": int(len(c)),
        "avg_value": float(np.mean([embedding[d] for d in c])),
        "max_value": float(np.max([embedding[d] for d in c])),
        "max_dim": int(c[np.argmax([embedding[d] for d in c])])
    } for i, c in enumerate(clusters)] [cite: 54, 55]

    top_indices = sorted(range(len(embedding)), key=lambda i: abs(embedding[i]), reverse=True)[:10] [cite: 56]
    top_dimensions = [{"dimension": int(idx), "value": float(embedding[idx])} for idx in top_indices] [cite: 56]

    return {"metrics": metrics, "clusters": cluster_info, "top_dimensions": top_dimensions} [cite: 56]

# ENHANCED: Function updated to accept and use target_keyword
def create_report_pdf(embedding, analysis, claude_analysis, business_type, page_type, target_keyword):
    """Create a PDF report with analysis, visualizations, and full context."""
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=0.5*inch, leftMargin=0.5*inch, topMargin=0.5*inch, bottomMargin=0.5*inch) [cite: 61]
        styles = getSampleStyleSheet() [cite: 61]
        title_style, heading1_style, heading2_style = styles['Title'], styles['Heading1'], styles['Heading2'] [cite: 62]
        normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], spaceBefore=6, spaceAfter=6, leading=14) [cite: 62, 63]
        story = []

        story.append(Paragraph("SEO Embedding Analysis Report", title_style)) [cite: 63]
        story.append(Spacer(1, 24))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style)) [cite: 64]
        story.append(Spacer(1, 12))

        # Add content categorization information including the keyword
        category_style = ParagraphStyle('CategoryStyle', parent=normal_style, fontName='Helvetica-Bold', textColor=colors.darkblue, fontSize=11) [cite: 65]
        business_display = business_type.replace("_", " ").title() [cite: 66]
        page_display = page_type.replace("_", " ").title() [cite: 67]
        story.append(Paragraph(f"Business Type: {business_display}", category_style)) [cite: 67]
        story.append(Paragraph(f"Page Type: {page_display}", category_style)) [cite: 67]
        if target_keyword:
            story.append(Paragraph(f"Target Keyword: {target_keyword}", category_style))
        story.append(Spacer(1, 24))

        # Metrics Table
        story.append(Paragraph("Key Metrics", heading1_style))
        metrics = analysis["metrics"]
        metrics_data = [
            ["Metric", "Value"],
            ["Dimensions", f"{metrics['dimension_count']}"],
            ["Mean Value", f"{metrics['mean_value']:.6f}"],
            ["Std Dev", f"{metrics['std_dev']:.6f}"],
            ["Min Value", f"{metrics['min_value']:.6f} (dim {metrics['min_dimension']})"],
            ["Max Value", f"{metrics['max_value']:.6f} (dim {metrics['max_dimension']})"],
            ["Significant Dims (>0.1)", f"{metrics['significant_dims']}"]
        ] [cite: 68, 69]
        metrics_table = Table(metrics_data, colWidths=[2*inch, 3.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ])) [cite: 70, 71, 72]
        story.append(metrics_table)
        story.append(PageBreak())

        # Visualizations
        story.append(Paragraph("Embedding Visualizations", heading1_style))
        def add_figure(fig_func, *args):
            fig = fig_func(*args)
            img_buffer = BytesIO()
            fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight') [cite: 75, 76, 77, 78, 79]
            plt.close(fig)
            img_buffer.seek(0)
            img = Image(img_buffer, width=7*inch, height=4*inch) [cite: 74]
            story.append(img)
            story.append(Spacer(1, 16))
        
        add_figure(plot_embedding_overview, embedding)
        add_figure(plot_top_dimensions, embedding)
        story.append(PageBreak())
        add_figure(plot_dimension_clusters, embedding)
        add_figure(plot_pca, embedding)

        # Claude Analysis
        story.append(PageBreak())
        story.append(Paragraph("Comprehensive Analysis", heading1_style)) [cite: 80]
        analysis_text = re.sub(r'<[^>]*>', '', claude_analysis) [cite: 82]
        for paragraph in analysis_text.split('\n\n'):
            if paragraph.strip():
                if paragraph.startswith('##'):
                    story.append(Paragraph(paragraph.replace('##', '').strip(), heading2_style)) [cite: 83]
                else:
                    story.append(Paragraph(paragraph.strip(), normal_style))
        
        doc.build(story) [cite: 86]
        pdf_value = buffer.getvalue()
        buffer.close()
        return pdf_value

    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

# MAIN APPLICATION LOGIC
def main():
    st.title("SEO Embedding Analysis Tool") [cite: 100]
    st.header("Content Input") [cite: 100]

    input_method = st.radio("Choose input method:", ("Paste Content", "Fetch from URL"), key="input_method_radio") [cite: 100]

    analyze_button_clicked = False
    if input_method == "Paste Content":
        st.session_state.content = st.text_area("Paste your content here:", value=st.session_state.get('content', ''), height=300, key="pasted_content_area") [cite: 101]
        if st.button("Analyze Pasted Content"):
            if st.session_state.content:
                analyze_button_clicked = True
            else:
                st.warning("Please paste content to analyze.")
    else: # Fetch from URL
        st.session_state.url = st.text_input("Enter the URL:", value=st.session_state.get('url', ''), placeholder="e.g., https://www.example.com/your-page") [cite: 102, 103]
        if st.button("Fetch and Analyze URL"):
            if st.session_state.url:
                with st.spinner(f"Fetching content from {st.session_state.url}..."):
                    fetched_content = fetch_content_from_url(st.session_state.url) [cite: 104]
                    if fetched_content:
                        st.session_state.content = fetched_content [cite: 105]
                        st.success("Content fetched successfully! Starting analysis...") [cite: 105]
                        analyze_button_clicked = True
                    else:
                        st.error("Failed to fetch content. Please check the URL or try pasting.") [cite: 106]
                        st.session_state.content = "" [cite: 107]
            else:
                st.warning("Please enter a URL.")

    st.write("### Content Categorization")
    st.write("Categorize your content and provide a target keyword for tailored recommendations:")

    # ENHANCED: 3-column layout for business type, page type, and keyword
    cat_col1, cat_col2, cat_col3 = st.columns(3)

    with cat_col1:
        business_type_display = st.selectbox("Business Type:", options=["Lead Generation/Service", "E-commerce", "SaaS/Tech", "Educational/Informational", "Local Business"], key="business_type_selectbox") [cite: 113]
        st.session_state.business_type = business_type_display.lower().replace(" ", "_").replace("/", "_") [cite: 115]

    with cat_col2:
        page_options = {
            "lead_generation_service": ["Landing Page", "Service Page", "Blog Post", "About Us", "Contact Page"],
            "ecommerce": ["Product Page", "Category Page", "Homepage", "Blog Post", "Checkout Page"],
            "saas_tech": ["Feature Page", "Pricing Page", "Homepage", "Blog Post", "Documentation"],
            "educational_informational": ["Course Page", "Resource Page", "Blog Post", "Homepage", "About Us"],
            "local_business": ["Homepage", "Service Page", "Location Page", "About Us", "Contact Page"]
        }.get(st.session_state.business_type, ["Homepage", "Blog Post"]) [cite: 116, 117, 118]
        page_type_display = st.selectbox("Page Type:", options=page_options, key="page_type_selectbox") [cite: 118, 119]
        st.session_state.page_type = page_type_display.lower().replace(" ", "_") [cite: 119]
    
    # NEW: Column for Target Keyword input
    with cat_col3:
        st.session_state.target_keyword = st.text_input(
            "Target Keyword (Optional):",
            value=st.session_state.get('target_keyword', ''),
            placeholder="e.g., 'data science jobs'",
            help="Provide the main keyword this content is targeting."
        )

    if analyze_button_clicked:
        if not st.session_state.google_api_key or not st.session_state.anthropic_api_key:
            st.error("Please provide both Google and Anthropic API keys in the sidebar.") [cite: 120, 121]
        else:
            with st.spinner("Analyzing content... This may take a minute."):
                st.session_state.embedding = get_embedding(st.session_state.content) [cite: 122]
                st.session_state.analysis = analyze_embedding(st.session_state.embedding) [cite: 123]
                # UPDATED: Pass the target keyword to the analysis function
                st.session_state.claude_analysis = analyze_with_claude(
                    st.session_state.embedding,
                    st.session_state.content,
                    st.session_state.business_type,
                    st.session_state.page_type,
                    st.session_state.target_keyword
                ) [cite: 124, 125]
                st.session_state.pdf_data = None
            st.success("Analysis complete!")
            # Use RerunException to force immediate re-render to show results
            raise RerunException(RerunData()) [cite: 126]

    if st.session_state.get('claude_analysis'):
        tab1, tab2, tab3, tab4 = st.tabs(["Visualizations", "Metrics", "Clusters", "Analysis Report"]) [cite: 127]

        with tab1: # Visualizations
            st.subheader("Embedding Overview")
            st.pyplot(plot_embedding_overview(st.session_state.embedding)) [cite: 128]
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top Dimensions")
                st.pyplot(plot_top_dimensions(st.session_state.embedding)) [cite: 129]
            with col2:
                st.subheader("Activation Distribution")
                st.pyplot(plot_activation_histogram(st.session_state.embedding)) [cite: 130]
            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Dimension Clusters")
                st.pyplot(plot_dimension_clusters(st.session_state.embedding)) [cite: 131]
            with col4:
                st.subheader("PCA Visualization")
                st.pyplot(plot_pca(st.session_state.embedding)) [cite: 132]

        with tab2: # Metrics
            st.subheader("Key Metrics")
            metrics = st.session_state.analysis["metrics"]
            col1, col2, col3 = st.columns(3) [cite: 133]
            with col1:
                st.metric("Dimensions", metrics["dimension_count"]) [cite: 133]
                st.metric("Mean Value", f"{metrics['mean_value']:.6f}") [cite: 133]
                st.metric("Std Dev", f"{metrics['std_dev']:.6f}") [cite: 133]
            with col2:
                st.metric("Min Value", f"{metrics['min_value']:.6f} (dim {metrics['min_dimension']})") [cite: 134]
                st.metric("Max Value", f"{metrics['max_value']:.6f} (dim {metrics['max_dimension']})") [cite: 134]
                st.metric("Median Value", f"{metrics['median_value']:.6f}") [cite: 134]
            with col3:
                st.metric("Positive Values", f"{metrics['positive_count']}/{metrics['dimension_count']}") [cite: 134]
                st.metric("Negative Values", f"{metrics['negative_count']}/{metrics['dimension_count']}") [cite: 135]
                st.metric("Significant Dims (>0.1)", f"{metrics['significant_dims']}") [cite: 135]

        with tab3: # Clusters
            st.subheader("Dimension Clusters")
            if not st.session_state.analysis["clusters"]:
                st.info("No significant dimension clusters detected.") [cite: 135]
            else:
                for cluster in st.session_state.analysis["clusters"]:
                    with st.expander(f"Cluster #{cluster['id']}: Dims {cluster['start_dim']}-{cluster['end_dim']}"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Size", f"{cluster['size']} dims") [cite: 136]
                        col2.metric("Avg Value", f"{cluster['avg_value']:.6f}") [cite: 137]
                        col3.metric("Max Value", f"{cluster['max_value']:.6f} (dim {cluster['max_dim']})") [cite: 137]

        with tab4: # Analysis Report
            st.subheader("Comprehensive Analysis Report")
            # Display context including the keyword
            context_html = f"""
            <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
                <p style="margin: 0; font-weight: bold;">Analysis tailored for:</p>
                <p style="margin: 0;"><strong>Business Type:</strong> {st.session_state.business_type.replace('_', ' ').title()}</p>
                <p style="margin: 0;"><strong>Page Type:</strong> {st.session_state.page_type.replace('_', ' ').title()}</p>
            """ [cite: 139, 140, 141]
            if st.session_state.target_keyword:
                context_html += f'<p style="margin: 0;"><strong>Target Keyword:</strong> {st.session_state.target_keyword}</p>'
            context_html += "</div>"
            st.markdown(context_html, unsafe_allow_html=True)
            
            st.markdown(st.session_state.claude_analysis, unsafe_allow_html=True) [cite: 142]

            st.write("---")
            st.subheader("Download Options")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(label="Download as Text", data=st.session_state.claude_analysis, file_name="seo_embedding_analysis.txt", mime="text/plain") [cite: 144, 145]
            with col2:
                if st.session_state.pdf_data is None:
                    if st.button("Generate PDF Report"):
                        with st.spinner("Generating PDF..."):
                            # UPDATED: Pass the target keyword to the PDF creation function
                            pdf_bytes = create_report_pdf(
                                st.session_state.embedding, st.session_state.analysis, st.session_state.claude_analysis,
                                st.session_state.business_type, st.session_state.page_type, st.session_state.target_keyword
                            ) [cite: 147, 148, 149, 150]
                            if pdf_bytes:
                                st.session_state.pdf_data = pdf_bytes [cite: 150]
                                st.success("PDF generated!")
                                raise RerunException(RerunData()) [cite: 152]
                else:
                    st.download_button(label="Download PDF Report", data=st.session_state.pdf_data, file_name="seo_embedding_analysis_report.pdf", mime="application/pdf") [cite: 154, 155, 156]
                    if st.button("Discard PDF"):
                        st.session_state.pdf_data = None [cite: 156]
                        raise RerunException(RerunData()) [cite: 157]

if __name__ == "__main__":
    try:
        main() [cite: 158]
    except RerunException as e:
        # This is expected, so we just re-raise it
        raise e
    except Exception as e:
        # Log other exceptions for debugging
        st.error(f"An unexpected error occurred in the main application flow: {e}") [cite: 158]
