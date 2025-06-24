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
from bs4 import BeautifulSoup # Import BeautifulSoup
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
from streamlit.runtime.scriptrunner import RerunData, RerunException # Import RerunData


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

def analyze_with_claude(embedding_data, content_snippet, business_type, page_type):
    """Get analysis from Claude with business and page type context"""
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

        message = anthropic_client.messages.create(
            model=current_settings["model"],
            max_tokens=current_settings["max_tokens"],
            temperature=current_settings["temperature"],
            thinking={
                "type": "enabled",
                "budget_tokens": current_settings["thinking_tokens"]
            },
            system=f"""You are an advanced SEO and NLP Embedding Analysis Expert with deep expertise in semantic content optimization, machine learning-driven content strategy, and advanced natural language processing techniques.

Your mission is to provide a comprehensive, multi-dimensional analysis of embedding data that transforms raw numerical information into actionable SEO and content strategy insights.

IMPORTANT CONTEXT: The content being analyzed is for {business_context}. Specifically, it is {page_context}. Tailor all your analysis and recommendations to this specific business and page type.

## ANALYTICAL METHODOLOGY
To ensure consistent analysis across different content types:

1. **Dimension Analysis Method**:
   - First identify the top 20 dimensions by absolute activation magnitude
   - Cluster these dimensions into 3-7 related groups based on activation patterns
   - Interpret what each cluster likely represents based on the specific content being analyzed
   - Never assign predetermined meanings to specific dimension ranges
   - Base all interpretations only on patterns present in the current embedding

2. **Statistical Consistency**:
   - Always flag dimensions with magnitude >0.1 as significant
   - Consider clusters of 3+ adjacent dimensions with similar activations as coherent topics
   - Identify semantic gaps as dimension ranges with consistently low activation ( tags
    formatted_text = re.sub(r'^###?\s*(.+)$', r'\1', analysis_text, flags=re.MULTILINE)
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
            "negative_count": int(np.sum(embedding  0.1))
        }

        # Find activation clusters
        significant_threshold = 0.1
        significant_dims = np.where(abs_embedding > significant_threshold)[0]

        # Find clusters (dimensions that are close to each other)
        clusters = []
        if len(significant_dims) > 0:
            current_cluster = [int(significant_dims[0])]  # Convert to int

            for i in range(1, len(significant_dims)):
                if significant_dims[i] - significant_dims[i-1]  0:
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

        story.append(Paragraph(f"Analysis tailored for: {business_display} | {page_display}", context_style))
        story.append(Spacer(1, 12))

        # Process Claude analysis text - convert from HTML/markdown to plain text
        # Remove HTML tags
        analysis_text = re.sub(r']*>', '', claude_analysis)

        # Function to process headings better
        def process_heading(text, style):
            text = text.strip()
            if text.startswith('##'):
                return Paragraph(text.replace('##', '').strip(), heading1_style)
            elif text.startswith('#'):
                return Paragraph(text.replace('#', '').strip(), heading2_style)
            elif text.strip() and text.strip()[0].isupper():  # Likely a section title without # marks
                first_line = text.split("\n")[0].strip()
                if len(first_line) 0.1)")

        # Tab 3: Clusters
        with tab3:
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

        # Tab 4: Claude Analysis with improved PDF generation and download
        with tab4:
            # Show content categorization info
            business_display = {
                "lead_generation": "Lead Generation/Service",
                "ecommerce": "E-commerce",
                "saas": "SaaS/Tech",
                "educational": "Educational/Informational",
                "local_business": "Local Business"
            }.get(st.session_state.business_type, st.session_state.business_type.replace("_", " ").title())

            page_display = st.session_state.page_type.replace("_", " ").title()

            st.markdown(f"""
            
                Analysis tailored for:
                Business Type: {business_display}
                Page Type: {page_display}
            
            """, unsafe_allow_html=True)

            st.subheader("Comprehensive Embedding Analysis Report")

            # Display the analysis content
            st.markdown(st.session_state.claude_analysis, unsafe_allow_html=True)

            # Download options
            st.write("---")

            with st.container():
                st.markdown("""
                
                Download Options
                
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                # Text download button
                with col1:
                    st.download_button(
                        label="Download as Text",
                        data=st.session_state.claude_analysis,
                        file_name="seo_embedding_analysis_report.txt",
                        mime="text/plain",
                        help="Download the analysis as a plain text file",
                        key="download_text_button"
                    )

                # PDF generation button and download
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
                                    # Trigger a rerun to show the download button immediately
                                    raise RerunException(RerunData()) # Use the correct rerun method with RerunData
                                except Exception as e:
                                    st.error(f"Error generating PDF: {str(e)}")
                    else:
                         # Show PDF download button if PDF has been generated
                         b64_pdf = base64.b64encode(st.session_state.pdf_data).decode('utf-8')
                         download_link = f'DOWNLOAD COMPLETE PDF REPORT'
                         st.markdown(download_link, unsafe_allow_html=True)

                         if st.button("Discard PDF and Generate Again", key="discard_pdf_button_tab"):
                            st.session_state.pdf_data = None
                            st.session_state.pdf_generated = False
                            # Trigger a rerun to update the button state
                            raise RerunException(RerunData()) # Use the correct rerun method with RerunData

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        # Add a basic content area to ensure something is visible
        st.write("## SEO Embedding Analysis Tool")
        st.write("This tool analyzes content semantics using embeddings.")

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/54161586/5420b228-c181-4707-922a-9035d23eb1eb/paste.txt
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/54161586/9ed21957-daa3-4d22-bbba-cf6901767961/paste.txt
