import streamlit as st
import pandas as pd
import json
import tempfile
import zipfile
from pathlib import Path
import io
import sys
import os

# Import the processing functions from main.py
from main import (
    process_xlsx, process_csv, process_sheet,
    bq_schema_from_df, format_dates_for_csv,
    write_clean_csv, find_unbalanced_quote_lines,
    infer_column, simple_header, strip_cell
)

# ============================================================================
# PAGE CONFIGURATION (Must be first Streamlit command)
# ============================================================================
st.set_page_config(
    page_title="BigQuery Data Processor",
    page_icon="assets/web/icons8-hub-pulsar-gradient-32.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# MAINTENANCE MODE CHECK
# ============================================================================
def is_maintenance_mode():
    """
    Check if maintenance mode is enabled via environment variable or Streamlit secret.
    Returns True if maintenance mode is active, False otherwise.
    """
    # Check environment variable first
    env_flag = os.getenv("MAINTENANCE_MODE", "").lower()
    if env_flag in ("true", "1", "yes", "on"):
        return True
    
    # Check Streamlit secrets as fallback
    try:
        secret_flag = st.secrets.get("MAINTENANCE_MODE", "").lower()
        if secret_flag in ("true", "1", "yes", "on"):
            return True
    except (AttributeError, KeyError, FileNotFoundError):
        # Secrets not configured, which is fine
        pass
    
    return False

# ============================================================================
# SHARED UI ELEMENTS
# ============================================================================
def render_shared_css():
    """Render the shared CSS styling used by both main app and maintenance page."""
st.markdown("""
<style>
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Hide image enlarge/fullscreen button - Only for logo images */
        div[data-testid="column"]:nth-of-type(2) [data-testid="stImage"] button,
        div[data-testid="column"]:nth-of-type(2) [data-testid="stImage"] > div > button,
        div[data-testid="column"]:nth-of-type(2) [data-testid="stImage"] > div > div > button,
        div[data-testid="column"]:nth-of-type(2) [data-testid="stImage"] * button,
        div[data-testid="column"]:nth-of-type(2) button[title*="Fullscreen"],
        div[data-testid="column"]:nth-of-type(2) button[aria-label*="Fullscreen"],
        div[data-testid="column"]:nth-of-type(2) button[aria-label*="fullscreen"],
        div[data-testid="column"]:nth-of-type(2) button[aria-label*="Expand"],
        div[data-testid="column"]:nth-of-type(2) button[aria-label*="expand"],
        .logo-wrapper [data-testid="stImage"] button,
        .logo-wrapper [data-testid="stImage"] > div > button,
        .logo-wrapper [data-testid="stImage"] > div > div > button,
        .logo-wrapper [data-testid="stImage"] * button,
        div[data-testid="column"]:nth-of-type(2):hover button,
        div[data-testid="column"]:nth-of-type(2) [data-testid="stImage"]:hover button {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            pointer-events: none !important;
            width: 0 !important;
            height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            position: absolute !important;
            left: -9999px !important;
        }
        
        /* Hide sidebar completely */
        section[data-testid="stSidebar"],
        div[data-testid="stSidebar"],
        .css-1d391kg,
        [data-testid="stSidebar"] {
            display: none !important;
            visibility: hidden !important;
        }
        
        /* Main container styling - Full width Hub style */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 100%;
            padding-left: 4rem;
            padding-right: 4rem;
            background-color: #ffffff;
        }
        
        @media (max-width: 768px) {
            .main .block-container {
                padding-left: 1.5rem;
                padding-right: 1.5rem;
            }
        }
        
        /* Ensure body has light background */
        body {
            background-color: #f8f9fa;
        }
        
        /* Main app background */
        .main {
            background-color: #f8f9fa;
        }
        
        /* Logo container - perfectly centered */
        .logo-wrapper {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            margin: 2rem auto 1.5rem auto;
            padding: 0;
            text-align: center;
        }
        
        /* Force logo image centering */
        .logo-wrapper img,
        .logo-wrapper [data-testid="stImage"] img,
        .logo-wrapper [data-testid="stImage"] {
            max-width: 280px !important;
            width: 280px !important;
            height: auto !important;
            margin: 0 auto !important;
            display: block !important;
            text-align: center !important;
        }
        
        /* Center the image container */
        [data-testid="stImage"],
        [data-testid="stImage"] > img,
        [data-testid="stImage"] > div {
            margin: 0 auto !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            text-align: center !important;
        }
        
        /* Center columns containing logo */
        .stColumn:has([data-testid="stImage"]),
        div[data-testid="column"]:has([data-testid="stImage"]) {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            text-align: center !important;
        }
        
        /* Ensure middle column centers content */
        div[data-testid="column"]:nth-of-type(2),
        div[data-testid="column"]:nth-of-type(2) > div,
        .stColumn:nth-of-type(2),
        .stColumn:nth-of-type(2) > div {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            text-align: center !important;
            width: 100% !important;
            margin: 0 auto !important;
        }
        
        /* Force center any image in the middle column */
        div[data-testid="column"]:nth-of-type(2) [data-testid="stImage"],
        div[data-testid="column"]:nth-of-type(2) [data-testid="stImage"] > div,
        div[data-testid="column"]:nth-of-type(2) [data-testid="stImage"] img,
        .stColumn:nth-of-type(2) [data-testid="stImage"],
        .stColumn:nth-of-type(2) [data-testid="stImage"] > div,
        .stColumn:nth-of-type(2) [data-testid="stImage"] img {
            margin: 0 auto !important;
            display: block !important;
            text-align: center !important;
            max-width: 280px !important;
            width: 280px !important;
        }
        
    .main-header {
        font-size: 2.5rem;
            font-weight: 600;
            color: #1a1a1a;
        text-align: center;
            margin: 0.5rem 0 0.75rem 0;
            letter-spacing: -0.02em;
            line-height: 1.2;
        }
        
        .sub-header {
            text-align: center;
            color: #6b7280;
            font-size: 1rem;
            margin-bottom: 2.5rem;
            font-weight: 400;
            line-height: 1.5;
            max-width: 100%;
        }
        
        /* Button styling - Hub style */
        .stButton > button,
        .stDownloadButton > button {
            background: #274156 !important;
            background-color: #274156 !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 500 !important;
            font-size: 0.9375rem !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
            width: 100% !important;
        }
        
        .stButton > button:hover,
        .stButton > button:focus,
        .stButton > button:active,
        .stDownloadButton > button:hover,
        .stDownloadButton > button:focus,
        .stDownloadButton > button:active {
            background: #335169 !important;
            background-color: #335169 !important;
            color: white !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }
        
        /* Maintenance page specific styles */
        .maintenance-container {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 4rem 3rem;
            margin: 3rem auto;
            max-width: 800px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .maintenance-title {
            font-size: 2rem;
            font-weight: 600;
            color: #111827;
            margin-bottom: 1.5rem;
        }
        
        .maintenance-message {
            font-size: 1.1rem;
            color: #6b7280;
            line-height: 1.7;
        margin-bottom: 2rem;
    }
        
        /* Footer styling */
        .footer {
            text-align: center;
            color: #6b7280;
            padding: 2rem 0;
            margin-top: 4rem;
            border-top: 1px solid #e5e7eb;
            font-size: 0.875rem;
        }
        
        /* Ensure all text is visible */
        .main {
            color: #374151;
        }
        
        .stMarkdown {
            color: #374151;
        }
        
        /* Feature cards */
        .feature-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            border: 1px solid #e9ecef;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .feature-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        /* Upload area styling */
        .upload-section {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border: 2px dashed #177091;
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            margin: 2rem 0;
            transition: all 0.3s ease;
        }
        
        .upload-section:hover {
            border-color: #13718F;
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        }
        
        /* Metric cards */
        .metric-container {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        /* Success/Warning/Error boxes */
    .success-box {
            background: #ecfdf5;
            border: 1px solid #10b981;
            border-left: 4px solid #10b981;
            border-radius: 10px;
            padding: 1.25rem;
        margin: 1rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
        
    .warning-box {
            background: #fffbeb;
            border: 1px solid #f59e0b;
            border-left: 4px solid #f59e0b;
            border-radius: 10px;
            padding: 1.25rem;
        margin: 1rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
        
    .error-box {
            background: #fef2f2;
            border: 1px solid #ef4444;
            border-left: 4px solid #ef4444;
            border-radius: 10px;
            padding: 1.25rem;
        margin: 1rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        /* Important notice box - translucent red/orange */
        .important-notice {
            background: rgba(220, 38, 38, 0.1);
            border: 1px solid rgba(220, 38, 38, 0.3);
            border-left: 4px solid rgba(220, 38, 38, 0.8);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 2rem 0;
            box-shadow: 0 2px 8px rgba(220, 38, 38, 0.1);
        }
        
        .important-notice h3 {
            color: #991b1b;
            font-weight: 600;
            margin-top: 0;
            margin-bottom: 1rem;
            font-size: 1.25rem;
        }
        
        .important-notice ul {
            color: #7f1d1d;
            margin: 0;
            padding-left: 1.5rem;
        }
        
        .important-notice ul li {
            margin-bottom: 0.75rem;
            line-height: 1.6;
        }
        
        .important-notice strong {
            color: #991b1b;
            font-weight: 600;
        }
        
        /* Section headers */
        h2 {
            color: #111827;
            font-weight: 600;
            margin-top: 2.5rem;
            margin-bottom: 1.25rem;
            font-size: 1.5rem;
            letter-spacing: -0.01em;
            scroll-margin-top: 2rem;
        }
        
        h3 {
            color: #111827;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            font-size: 1.25rem;
            letter-spacing: -0.01em;
            scroll-margin-top: 2rem;
        }
        
        /* Smooth scrolling */
        html {
            scroll-behavior: smooth;
        }
        
        /* Better spacing for sections */
        .stMarkdown > div {
            margin-bottom: 1.5rem;
        }
        
        /* Improved schema review table styling */
        .schema-review-row {
            transition: all 0.2s ease;
            border-radius: 8px;
            padding: 0.25rem 0;
            margin-bottom: 0.25rem;
        }
        
        .schema-review-row:hover {
            background-color: #f9fafb;
            transform: translateX(4px);
        }
        
        /* Reduce spacing between schema review rows */
        .schema-review-row + .schema-review-row {
            margin-top: 0;
        }
        
        /* Reduce column padding within schema review rows */
        .schema-review-row div[data-testid="column"] {
            padding: 0.25rem 0.5rem !important;
        }
        
        /* Better selectbox styling */
        [data-baseweb="select"] {
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
        }
        
        [data-baseweb="select"]:hover {
            border-color: #274156 !important;
            box-shadow: 0 0 0 3px rgba(39, 65, 86, 0.1) !important;
        }
        
        [data-baseweb="select"]:focus-within {
            border-color: #274156 !important;
            box-shadow: 0 0 0 3px rgba(39, 65, 86, 0.15) !important;
        }
        
        /* Improved file info card */
        .file-info {
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .file-info:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        /* Better expander styling */
        [data-testid="stExpander"] {
            border: 1px solid #e5e7eb !important;
            border-radius: 12px !important;
            margin-bottom: 1rem !important;
            transition: all 0.2s ease !important;
        }
        
        [data-testid="stExpander"]:hover {
            border-color: #274156 !important;
            box-shadow: 0 2px 8px rgba(39, 65, 86, 0.1) !important;
        }
        
        /* Default column padding */
        div[data-testid="column"] {
            padding: 0.5rem !important;
        }
        
        /* Reduced spacing for columns in schema review */
        .schema-review-container div[data-testid="column"],
        .schema-review-row div[data-testid="column"] {
            padding: 0.25rem 0.5rem !important;
        }
        
        /* Better visual hierarchy for schema review */
        .schema-column-name {
            font-weight: 600;
            color: #111827;
            font-size: 0.95rem;
        }
        
        .schema-inferred-type {
            color: #6b7280;
            font-size: 0.9rem;
            padding: 0.25rem 0.5rem;
            background: #f3f4f6;
            border-radius: 6px;
            display: inline-block;
        }
        
        .schema-sample-values {
            color: #6b7280;
            font-size: 0.85rem;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            background: #f9fafb;
            padding: 0.5rem;
            border-radius: 6px;
            line-height: 1.5;
        }
        
        /* Loading spinner improvements */
        .stSpinner > div {
            border-color: #274156 !important;
        }
        
        /* Better button focus states */
        .stButton > button:focus,
        .stDownloadButton > button:focus {
            outline: 3px solid rgba(39, 65, 86, 0.3) !important;
            outline-offset: 2px !important;
        }
        
        /* Protect navigation buttons from being hidden */
        button[data-testid*="nav_to_schema"],
        button[data-testid*="nav_to_docs"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
            position: relative !important;
        }
        
        /* Ensure navigation buttons are always visible on hover */
        button:hover[data-testid*="nav_to_schema"],
        button:hover[data-testid*="nav_to_docs"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
        }
        
        /* Additional protection for navigation button containers */
        .stButton:has(button[data-testid*="nav_to_schema"]),
        .stButton:has(button[data-testid*="nav_to_docs"]) {
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
        }
        
        /* Protect download buttons from being hidden */
        .stDownloadButton > button,
        button[data-testid*="download"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
            position: relative !important;
        }
        
        /* Ensure download buttons are always visible on hover */
        .stDownloadButton > button:hover,
        button:hover[data-testid*="download"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
        }
        
        /* Additional protection for download button containers */
        .stDownloadButton {
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
        }
        
        /* Improved metric containers */
        .metric-container {
            transition: all 0.2s ease;
        }
        
        .metric-container:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        }
        
        /* Better success/warning/error boxes */
        .success-box, .warning-box, .error-box {
            animation: slideIn 0.3s ease-out;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Improved info cards */
        .info-card {
            transition: all 0.2s ease;
        }
        
        .info-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
        }
        
        /* Better divider styling */
        hr {
            border: none;
            border-top: 1px solid #e5e7eb;
            margin: 2rem 0;
        }
        
        /* Improved spacing for file uploader */
        [data-testid="stFileUploader"] {
            margin: 1.5rem 0 !important;
        }
        
        /* Better table/card spacing */
        .schema-review-container {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            transition: all 0.2s ease;
        }
        
        .schema-review-container:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        /* Mobile responsiveness improvements */
        @media (max-width: 768px) {
            .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            
            h2 {
                font-size: 1.25rem;
                margin-top: 1.5rem;
            }
            
            .schema-review-container {
        padding: 1rem;
            }
        }
        
        /* Focus visible for accessibility */
        *:focus-visible {
            outline: 2px solid #274156;
            outline-offset: 2px;
            border-radius: 4px;
        }
        
        /* Better scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }
        
        /* Info text styling */
        .info-text {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 2rem;
            margin: 2rem 0;
            line-height: 1.7;
            color: #374151;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            max-width: 100%;
        }
        
        .info-text h3 {
            color: #111827 !important;
            margin-top: 0 !important;
            margin-bottom: 1.25rem !important;
            font-size: 1.25rem;
            font-weight: 600;
        }
        
        .info-text ul {
            color: #374151;
            margin: 0 0 1rem 0;
            padding-left: 1.5rem;
        }
        
        .info-text ul li {
            margin-bottom: 0.75rem;
            line-height: 1.6;
        }
        
        .info-text p {
            color: #374151;
            margin: 0;
        }
        
        .info-text strong {
            color: #111827;
            font-weight: 600;
        }
        
        /* File info styling */
        .file-info {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 1.25rem 1.5rem;
            margin: 1rem 0 1.5rem 0;
            color: #374151;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            font-size: 0.9375rem;
            line-height: 1.6;
            width: 100%;
            display: block;
        }
        
        .file-info strong {
            color: #111827;
            font-weight: 600;
        }
        
        /* Alert badge styling */
        .upload-alert {
            display: inline-flex;
            align-items: center;
            background: #fef3c7;
            border: 1px solid #f59e0b;
            border-radius: 6px;
            padding: 0.375rem 0.75rem;
            margin-left: 0.75rem;
            font-size: 0.875rem;
            color: #92400e;
            font-weight: 500;
        }
        
        .upload-alert::before {
            content: "ℹ️";
            margin-right: 0.375rem;
        }
        
        /* Fix text color in success/warning/error boxes */
        .success-box, .warning-box, .error-box {
            color: #374151;
            border-radius: 10px;
            border: 1px solid;
        }
        
        .success-box {
            border-color: #10b981;
        }
        
        .warning-box {
            border-color: #f59e0b;
        }
        
        .error-box {
            border-color: #ef4444;
        }
        
        .success-box strong, .warning-box strong, .error-box strong {
            color: #111827;
        }
        
        /* Metric container text */
        .metric-container {
            color: #374151;
            border: 1px solid #e5e7eb;
        }
        
        /* Download button styling */
        .stDownloadButton > button,
        .stDownloadButton button,
        button[data-testid="baseButton-secondary"],
        button[data-testid="baseButton-secondary"]:hover,
        button[data-testid="baseButton-secondary"]:focus,
        button[data-testid="baseButton-secondary"]:active,
        div[data-testid="stDownloadButton"] > button,
        div[data-testid="stDownloadButton"] button {
            background: #274156 !important;
            background-color: #274156 !important;
            color: #ffffff !important;
            border: none !important;
            border-color: transparent !important;
            border-radius: 8px !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 500 !important;
            font-size: 0.9375rem !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
            width: 100% !important;
        }
        
        .stDownloadButton > button:hover,
        .stDownloadButton button:hover,
        button[data-testid="baseButton-secondary"]:hover,
        div[data-testid="stDownloadButton"] > button:hover,
        div[data-testid="stDownloadButton"] button:hover {
            background: #335169 !important;
            background-color: #335169 !important;
            color: #ffffff !important;
            border: none !important;
            border-color: transparent !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }
        
        /* File uploader styling */
        [data-testid="stFileUploader"] {
            color: #000000 !important;
            background: transparent !important;
            background-color: transparent !important;
        }
        
        [data-testid="stFileUploader"] > div {
            background: transparent !important;
            background-color: transparent !important;
            border-color: #d1d5db !important;
        }
        
        [data-testid="stFileUploader"] > div > div {
            background: transparent !important;
            background-color: transparent !important;
            border-color: #d1d5db !important;
        }
        
        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] div,
        [data-testid="stFileUploader"] *:not(button) {
            color: #000000 !important;
        }
        
        [data-testid="stFileUploader"] > div:hover {
            background: transparent !important;
            background-color: transparent !important;
            border-color: #d1d5db !important;
        }
        
        [data-testid="stFileUploader"] > div > div:hover {
            background: transparent !important;
            background-color: transparent !important;
            border-color: #d1d5db !important;
        }
        
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploader"] > div > div > button,
        [data-testid="stFileUploader"] button[type="button"] {
            background: #274156 !important;
            background-color: #274156 !important;
            border-color: #274156 !important;
            border: 1px solid #274156 !important;
            color: #ffffff !important;
            border-radius: 8px !important;
        }
        
        [data-testid="stFileUploader"] button:hover,
        [data-testid="stFileUploader"] button:focus,
        [data-testid="stFileUploader"] button:active,
        [data-testid="stFileUploader"] > div > div > button:hover,
        [data-testid="stFileUploader"] > div > div > button:focus,
        [data-testid="stFileUploader"] > div > div > button:active,
        [data-testid="stFileUploader"] button[type="button"]:hover,
        [data-testid="stFileUploader"] button[type="button"]:focus {
            background: #335169 !important;
            background-color: #335169 !important;
            border-color: #335169 !important;
            border: 1px solid #335169 !important;
            color: #ffffff !important;
            outline: none !important;
        }
        
        /* Style the clear/remove button - Hide SVG and show "Clear" text */
        [data-testid="stFileUploader"] button[data-clear-text-added],
        [data-testid="stFileUploader"] button[aria-label*="Remove"],
        [data-testid="stFileUploader"] button[aria-label*="remove"],
        [data-testid="stFileUploader"] button[title*="Remove"],
        [data-testid="stFileUploader"] button[title*="remove"],
        [data-testid="stFileUploader"] button[aria-label*="Delete"],
        [data-testid="stFileUploader"] button[aria-label*="delete"],
        [data-testid="stFileUploader"] button[aria-label*="Clear"],
        [data-testid="stFileUploader"] button[aria-label*="clear"],
        [data-testid="stFileUploader"] button svg[data-testid="CloseIcon"],
        [data-testid="stFileUploader"] button:has(svg[data-testid="CloseIcon"]),
        [data-testid="stFileUploader"] button:has(svg path[d*="M18 6L6 18"]),
        [data-testid="stFileUploader"] button:has(svg path[d*="M6 6l12 12"]),
        [data-testid="stFileUploader"] button:has(svg path[d*="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"]),
        [data-testid="stFileUploader"] button:has(svg path[d*="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z"]),
        [data-testid="stFileUploader"] button[type="button"]:has(svg),
        [data-testid="stFileUploader"] > div > div > button:has(svg),
        [data-testid="stFileUploader"] > div > div > div > button:has(svg) {
            background: #274156 !important;
            background-color: #274156 !important;
            border: 1px solid #274156 !important;
            border-radius: 8px !important;
            height: 40px !important;
            min-height: 40px !important;
            max-height: 40px !important;
            padding: 0 12px !important;
            margin: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            color: #ffffff !important;
            font-weight: 500 !important;
            font-size: 0.9375rem !important;
            flex-shrink: 0 !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
            transition: all 0.2s ease !important;
        }
        
        /* When button has "Clear" text, make it wider */
        [data-testid="stFileUploader"] button[data-clear-text-added] {
            min-width: 60px !important;
            width: auto !important;
            max-width: none !important;
        }
        
        /* When button still has SVG (not converted yet), keep it square */
        [data-testid="stFileUploader"] button:has(svg):not([data-clear-text-added]) {
            width: 40px !important;
            min-width: 40px !important;
            max-width: 40px !important;
            padding: 0 !important;
        }
        
        /* Hide SVG in clear buttons */
        [data-testid="stFileUploader"] button[data-clear-text-added] svg,
        [data-testid="stFileUploader"] button[data-clear-text-added] > svg {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            width: 0 !important;
            height: 0 !important;
        }
        
        /* Additional targeting for clear button - any button with SVG that's small */
        [data-testid="stFileUploader"] > div > div > div button:has(svg),
        [data-testid="stFileUploader"] > div > div button:has(svg),
        [data-testid="stFileUploader"] ul button:has(svg),
        [data-testid="stFileUploader"] li button:has(svg) {
            min-width: 60px !important;
            width: auto !important;
            padding: 0 12px !important;
        }
        
        /* Target all SVG elements inside remove buttons */
        [data-testid="stFileUploader"] button[aria-label*="Remove"] svg,
        [data-testid="stFileUploader"] button[aria-label*="remove"] svg,
        [data-testid="stFileUploader"] button[title*="Remove"] svg,
        [data-testid="stFileUploader"] button[title*="remove"] svg,
        [data-testid="stFileUploader"] button[aria-label*="Delete"] svg,
        [data-testid="stFileUploader"] button[aria-label*="delete"] svg,
        [data-testid="stFileUploader"] button[aria-label*="Clear"] svg,
        [data-testid="stFileUploader"] button[aria-label*="clear"] svg,
        [data-testid="stFileUploader"] button svg[data-testid="CloseIcon"],
        [data-testid="stFileUploader"] button:has(svg[data-testid="CloseIcon"]) svg,
        [data-testid="stFileUploader"] button:has(svg path[d*="M18 6L6 18"]) svg,
        [data-testid="stFileUploader"] button:has(svg path[d*="M6 6l12 12"]) svg,
        [data-testid="stFileUploader"] button:has(svg path[d*="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"]) svg,
        [data-testid="stFileUploader"] button[type="button"]:has(svg) svg,
        [data-testid="stFileUploader"] > div > div > button:has(svg) svg,
        [data-testid="stFileUploader"] > div > div > div > button:has(svg) svg {
            color: #ffffff !important;
            fill: #ffffff !important;
            stroke: #ffffff !important;
            width: 20px !important;
            height: 20px !important;
        }
        
        /* Target all path elements inside SVG */
        [data-testid="stFileUploader"] button[aria-label*="Remove"] svg path,
        [data-testid="stFileUploader"] button[aria-label*="remove"] svg path,
        [data-testid="stFileUploader"] button[title*="Remove"] svg path,
        [data-testid="stFileUploader"] button[title*="remove"] svg path,
        [data-testid="stFileUploader"] button[aria-label*="Delete"] svg path,
        [data-testid="stFileUploader"] button[aria-label*="delete"] svg path,
        [data-testid="stFileUploader"] button[aria-label*="Clear"] svg path,
        [data-testid="stFileUploader"] button[aria-label*="clear"] svg path,
        [data-testid="stFileUploader"] button svg[data-testid="CloseIcon"] path,
        [data-testid="stFileUploader"] button:has(svg[data-testid="CloseIcon"]) svg path,
        [data-testid="stFileUploader"] button:has(svg path[d*="M18 6L6 18"]) svg path,
        [data-testid="stFileUploader"] button:has(svg path[d*="M6 6l12 12"]) svg path,
        [data-testid="stFileUploader"] button:has(svg path[d*="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"]) svg path,
        [data-testid="stFileUploader"] button[type="button"]:has(svg) svg path,
        [data-testid="stFileUploader"] > div > div > button:has(svg) svg path,
        [data-testid="stFileUploader"] > div > div > div > button:has(svg) svg path,
        [data-testid="stFileUploader"] button svg path {
            fill: #ffffff !important;
            stroke: #ffffff !important;
            color: #ffffff !important;
        }
        
        /* Hover states */
        [data-testid="stFileUploader"] button[data-clear-text-added]:hover,
        [data-testid="stFileUploader"] button[aria-label*="Remove"]:hover,
        [data-testid="stFileUploader"] button[aria-label*="remove"]:hover,
        [data-testid="stFileUploader"] button[title*="Remove"]:hover,
        [data-testid="stFileUploader"] button[title*="remove"]:hover,
        [data-testid="stFileUploader"] button[aria-label*="Delete"]:hover,
        [data-testid="stFileUploader"] button[aria-label*="delete"]:hover,
        [data-testid="stFileUploader"] button:has(svg[data-testid="CloseIcon"]):hover,
        [data-testid="stFileUploader"] button:has(svg path[d*="M18 6L6 18"]):hover,
        [data-testid="stFileUploader"] button:has(svg path[d*="M6 6l12 12"]):hover,
        [data-testid="stFileUploader"] button[type="button"]:has(svg):hover,
        [data-testid="stFileUploader"] > div > div > button:has(svg):hover {
            background: #335169 !important;
            background-color: #335169 !important;
            border-color: #335169 !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

def render_shared_logo_script():
    """Render the shared JavaScript for logo centering and button removal."""
    st.markdown("""
    <script>
    function centerLogo() {
        const middleColumn = document.querySelector('div[data-testid="column"]:nth-of-type(2)');
        if (middleColumn) {
            middleColumn.style.display = 'flex';
            middleColumn.style.justifyContent = 'center';
            middleColumn.style.alignItems = 'center';
            middleColumn.style.textAlign = 'center';
            middleColumn.style.width = '100%';
            middleColumn.style.margin = '0 auto';
            
            const logoImage = middleColumn.querySelector('[data-testid="stImage"]');
            if (logoImage) {
                logoImage.style.margin = '0 auto';
                logoImage.style.display = 'block';
                logoImage.style.textAlign = 'center';
                
                const nestedDivs = logoImage.querySelectorAll('div, img');
                nestedDivs.forEach(function(el) {
                    el.style.margin = '0 auto';
                    el.style.display = 'block';
                    el.style.textAlign = 'center';
                });
            }
        }
    }
    
    centerLogo();
    setTimeout(centerLogo, 50);
    setTimeout(centerLogo, 100);
    setTimeout(centerLogo, 200);
    setTimeout(centerLogo, 500);
    
    const centerObserver = new MutationObserver(centerLogo);
    centerObserver.observe(document.body, { childList: true, subtree: true });
    
    function removeLogoButtons() {
        const middleColumn = document.querySelector('div[data-testid="column"]:nth-of-type(2)');
        if (middleColumn) {
            const allButtons = middleColumn.querySelectorAll('button');
            allButtons.forEach(function(btn) {
                const isImageButton = btn.getAttribute('title') && btn.getAttribute('title').toLowerCase().includes('fullscreen') ||
                                     btn.getAttribute('aria-label') && btn.getAttribute('aria-label').toLowerCase().includes('fullscreen') ||
                                     btn.getAttribute('aria-label') && btn.getAttribute('aria-label').toLowerCase().includes('expand') ||
                                     btn.closest('[data-testid="stImage"]');
                
                if (isImageButton) {
                    // Don't hide navigation buttons
                    const btnText = (btn.textContent || '').trim();
                    const btnKey = btn.getAttribute('data-testid') || '';
                    if (btnText !== 'Schema Guide' && 
                        btnText !== 'BigQuery Guide' && 
                        btnKey !== 'nav_to_schema_docs' && 
                        btnKey !== 'nav_to_docs') {
                        btn.style.display = 'none';
                        btn.style.visibility = 'hidden';
                        btn.style.opacity = '0';
                    }
                    btn.style.pointerEvents = 'none';
                    btn.style.width = '0';
                    btn.style.height = '0';
                    btn.style.padding = '0';
                    btn.style.margin = '0';
                    btn.style.position = 'absolute';
                    btn.style.left = '-9999px';
                    try {
                        btn.remove();
                    } catch(e) {}
                }
            });
            
            const logoImages = middleColumn.querySelectorAll('[data-testid="stImage"]');
            logoImages.forEach(function(imgContainer) {
                const buttons = imgContainer.querySelectorAll('button');
                buttons.forEach(function(btn) {
                    // Don't hide navigation buttons
                    const btnText = (btn.textContent || '').trim();
                    const btnKey = btn.getAttribute('data-testid') || '';
                    if (btnText !== 'Schema Guide' && 
                        btnText !== 'BigQuery Guide' && 
                        btnKey !== 'nav_to_schema_docs' && 
                        btnKey !== 'nav_to_docs') {
                        btn.style.display = 'none';
                        btn.style.visibility = 'hidden';
                        btn.style.opacity = '0';
                    }
                    btn.style.pointerEvents = 'none';
                    btn.style.width = '0';
                    btn.style.height = '0';
                    btn.style.padding = '0';
                    btn.style.margin = '0';
                    btn.style.position = 'absolute';
                    btn.style.left = '-9999px';
                    try {
                        btn.remove();
                    } catch(e) {}
                });
            });
        }
    }
    
    removeLogoButtons();
    setTimeout(removeLogoButtons, 50);
    setTimeout(removeLogoButtons, 100);
    setTimeout(removeLogoButtons, 200);
    setTimeout(removeLogoButtons, 500);
    setInterval(removeLogoButtons, 200);
    
    const observer = new MutationObserver(function(mutations) {
        removeLogoButtons();
    });
    observer.observe(document.body, { 
        childList: true, 
        subtree: true,
        attributes: true,
        attributeFilter: ['style', 'class']
    });
    
    function styleFileUploaderClearButton() {
        // Find the file uploader - try multiple selectors
        const fileUploader = document.querySelector('[data-testid="stFileUploader"]') || 
                            document.querySelector('.stFileUploader') ||
                            document.querySelector('[class*="fileUploader"]');
        
        // Find all buttons - start with file uploader, but also search document-wide for small square buttons
        let allButtons = [];
        
        if (fileUploader) {
            allButtons = Array.from(fileUploader.querySelectorAll('button'));
            const parentContainer = fileUploader.closest('div');
            if (parentContainer) {
                const nearbyButtons = Array.from(parentContainer.querySelectorAll('button'));
                nearbyButtons.forEach(function(btn) {
                    if (!allButtons.includes(btn)) {
                        allButtons.push(btn);
                    }
                });
            }
        }
        
        // Also search document-wide for any small square button that might be the clear button
        // Look for buttons that are 40x40 or smaller with SVG, positioned near file names
        const allDocumentButtons = Array.from(document.querySelectorAll('button'));
        allDocumentButtons.forEach(function(btn) {
            const svg = btn.querySelector('svg');
            const btnText = (btn.textContent || '').trim();
            
            // Skip if already in our list or if it's the Browse button
            if (allButtons.includes(btn) || btnText.toLowerCase().includes('browse')) {
                return;
            }
            
            // If it's a small square button with SVG and no text, it might be the clear button
            if (svg && !btnText) {
                try {
                    const computedStyle = window.getComputedStyle(btn);
                    const btnWidth = parseFloat(computedStyle.width);
                    const btnHeight = parseFloat(computedStyle.height);
                    
                    // Check if it's a small square button (40x40 or smaller)
                    if (btnWidth <= 45 && btnHeight <= 45) {
                        // Check if it's near a file uploader or file list
                        const parent = btn.closest('[data-testid="stFileUploader"]') || 
                                      btn.closest('ul') || 
                                      btn.closest('li') ||
                                      btn.closest('[class*="file"]');
                        if (parent) {
                            allButtons.push(btn);
                        }
                    }
                } catch(e) {
                    // If we can't compute style, still check if it has remove/delete/clear attributes
                    const ariaLabel = (btn.getAttribute('aria-label') || '').toLowerCase();
                    const title = (btn.getAttribute('title') || '').toLowerCase();
                    if (ariaLabel.includes('remove') || ariaLabel.includes('delete') || ariaLabel.includes('clear') ||
                        title.includes('remove') || title.includes('delete') || title.includes('clear')) {
                        allButtons.push(btn);
                    }
                }
            }
        });
        
        allButtons.forEach(function(btn) {
            const btnText = (btn.textContent || '').trim();
            const ariaLabel = (btn.getAttribute('aria-label') || '').toLowerCase();
            const title = (btn.getAttribute('title') || '').toLowerCase();
            const svg = btn.querySelector('svg');
            const computedStyle = window.getComputedStyle(btn);
            const btnWidth = parseFloat(computedStyle.width);
            const btnHeight = parseFloat(computedStyle.height);
            
            // Skip the "Browse files" button - it has visible text like "Browse files"
            if (btnText && (btnText.toLowerCase().includes('browse') || btnText.toLowerCase().includes('file'))) {
                return;
            }
            
            // Skip if already converted to "Clear" button
            if (btn.hasAttribute('data-clear-text-added') && btnText === 'Clear') {
                // Ensure styling is still correct
                btn.style.setProperty('background', '#274156', 'important');
                btn.style.setProperty('background-color', '#274156', 'important');
                btn.style.setProperty('color', '#ffffff', 'important');
                btn.style.setProperty('min-width', '60px', 'important');
                btn.style.setProperty('width', 'auto', 'important');
                btn.style.setProperty('height', '40px', 'important');
                btn.style.setProperty('padding', '0 12px', 'important');
                return;
            }
            
            // More aggressive detection: 
            // 1. Has SVG but no text
            // 2. Has remove/delete/clear in aria-label/title
            // 3. Small square button (typically 40x40 or smaller) with SVG
            // 4. Button positioned next to file name
            const isSmallSquare = btnWidth <= 45 && btnHeight <= 45 && svg;
            const hasRemoveLabel = ariaLabel.includes('remove') || 
                                  ariaLabel.includes('delete') || 
                                  ariaLabel.includes('clear') ||
                                  title.includes('remove') || 
                                  title.includes('delete') || 
                                  title.includes('clear');
            const isClearButton = (svg && !btnText) || hasRemoveLabel || isSmallSquare;
            
            if (!isClearButton) {
                return; // Skip other buttons
            }
            
            // Remove all SVG elements and their parent containers
            const allSvgs = btn.querySelectorAll('svg');
            allSvgs.forEach(function(svgEl) {
                svgEl.remove();
            });
            
            // Remove any child elements that might be icon containers
            const children = Array.from(btn.children);
            children.forEach(function(child) {
                if (child.tagName === 'SVG' || child.querySelector('svg')) {
                    child.remove();
                }
            });
            
            // Clear all content and set "Clear" text
            btn.innerHTML = '';
            btn.textContent = 'Clear';
            btn.setAttribute('data-clear-text-added', 'true');
            
            // Apply comprehensive styling - use cssText for maximum override
            btn.style.cssText = 'background: #274156 !important; ' +
                               'background-color: #274156 !important; ' +
                               'border: 1px solid #274156 !important; ' +
                               'border-color: #274156 !important; ' +
                               'border-radius: 8px !important; ' +
                               'min-width: 60px !important; ' +
                               'width: auto !important; ' +
                               'max-width: none !important; ' +
                               'height: 40px !important; ' +
                               'min-height: 40px !important; ' +
                               'max-height: 40px !important; ' +
                               'padding: 0 12px !important; ' +
                               'margin: 0 !important; ' +
                               'display: flex !important; ' +
                               'align-items: center !important; ' +
                               'justify-content: center !important; ' +
                               'color: #ffffff !important; ' +
                               'font-weight: 500 !important; ' +
                               'font-size: 0.9375rem !important; ' +
                               'flex-shrink: 0 !important; ' +
                               'box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important; ' +
                               'transition: all 0.2s ease !important; ' +
                               'cursor: pointer !important;';
            
            // Add hover effect (only if not already added)
            if (!btn.hasAttribute('data-hover-added')) {
                btn.setAttribute('data-hover-added', 'true');
                const hoverEnter = function() {
                    this.style.setProperty('background', '#335169', 'important');
                    this.style.setProperty('background-color', '#335169', 'important');
                    this.style.setProperty('border-color', '#335169', 'important');
                    this.style.setProperty('box-shadow', '0 2px 4px rgba(0,0,0,0.1)', 'important');
                };
                const hoverLeave = function() {
                    this.style.setProperty('background', '#274156', 'important');
                    this.style.setProperty('background-color', '#274156', 'important');
                    this.style.setProperty('border-color', '#274156', 'important');
                    this.style.setProperty('box-shadow', '0 1px 2px rgba(0,0,0,0.05)', 'important');
                };
                btn.addEventListener('mouseenter', hoverEnter);
                btn.addEventListener('mouseleave', hoverLeave);
            }
        });
    }
    
    // Run immediately and more frequently
    styleFileUploaderClearButton();
    setTimeout(styleFileUploaderClearButton, 50);
    setTimeout(styleFileUploaderClearButton, 100);
    setTimeout(styleFileUploaderClearButton, 200);
    setTimeout(styleFileUploaderClearButton, 500);
    setTimeout(styleFileUploaderClearButton, 1000);
    setTimeout(styleFileUploaderClearButton, 2000);
    
    // Use setInterval for continuous monitoring
    setInterval(styleFileUploaderClearButton, 500);
    
    const fileUploaderObserver = new MutationObserver(function(mutations) {
        styleFileUploaderClearButton();
    });
    fileUploaderObserver.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['style', 'class', 'data-testid']
    });
    
    // Smooth scroll to sections
    function smoothScrollToElement(element) {
        if (element) {
            const headerOffset = 80;
            const elementPosition = element.getBoundingClientRect().top;
            const offsetPosition = elementPosition + window.pageYOffset - headerOffset;
            
            window.scrollTo({
                top: offsetPosition,
                behavior: 'smooth'
            });
        }
    }
    
    // Add smooth scroll to all anchor links
    document.addEventListener('click', function(e) {
        if (e.target.tagName === 'A' && e.target.getAttribute('href') && e.target.getAttribute('href').startsWith('#')) {
            e.preventDefault();
            const targetId = e.target.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            smoothScrollToElement(targetElement);
        }
    });
    
    // Highlight selected schema row on interaction
    function addSchemaRowInteractions() {
        const schemaRows = document.querySelectorAll('.schema-review-row');
        schemaRows.forEach(function(row) {
            // Only add listeners if not already added
            if (!row.dataset.interactive) {
                row.dataset.interactive = 'true';
                row.addEventListener('mouseenter', function() {
                    this.style.backgroundColor = '#f9fafb';
                    this.style.transform = 'translateX(4px)';
                });
                row.addEventListener('mouseleave', function() {
                    this.style.backgroundColor = '';
                    this.style.transform = '';
                });
            }
        });
    }
    
    // Run on page load and when DOM changes
    addSchemaRowInteractions();
    setTimeout(addSchemaRowInteractions, 100);
    setTimeout(addSchemaRowInteractions, 500);
    
    const schemaObserver = new MutationObserver(function(mutations) {
        addSchemaRowInteractions();
    });
    schemaObserver.observe(document.body, {
        childList: true,
        subtree: true
    });
    
    // Prevent any interference with navigation buttons
    function protectNavigationButtons() {
        // Find buttons by their data-testid keys
        const navButtons = document.querySelectorAll('button[data-testid*="nav_to_schema"], button[data-testid*="nav_to_docs"]');
        navButtons.forEach(function(button) {
            // Force button to always be visible and clickable
            button.style.setProperty('display', 'flex', 'important');
            button.style.setProperty('visibility', 'visible', 'important');
            button.style.setProperty('opacity', '1', 'important');
            button.style.setProperty('pointer-events', 'auto', 'important');
            button.style.setProperty('position', 'relative', 'important');
            
            // Also protect the parent container
            const parent = button.closest('.stButton');
            if (parent) {
                parent.style.setProperty('display', 'block', 'important');
                parent.style.setProperty('visibility', 'visible', 'important');
                parent.style.setProperty('opacity', '1', 'important');
            }
            
            // Prevent any hover effects that might hide it
            button.onmouseenter = function() {
                this.style.setProperty('display', 'flex', 'important');
                this.style.setProperty('visibility', 'visible', 'important');
                this.style.setProperty('opacity', '1', 'important');
            };
            button.onmouseleave = function() {
                this.style.setProperty('display', 'flex', 'important');
                this.style.setProperty('visibility', 'visible', 'important');
                this.style.setProperty('opacity', '1', 'important');
            };
            
            // Mark as protected
            if (!button.dataset.protected) {
                button.dataset.protected = 'true';
            }
        });
        
        // Also check by text content as fallback
        const allButtons = document.querySelectorAll('.stButton > button');
        allButtons.forEach(function(button) {
            const buttonText = (button.textContent || '').trim();
            if (buttonText === 'Schema Guide' || buttonText === 'BigQuery Guide') {
                button.style.setProperty('display', 'flex', 'important');
                button.style.setProperty('visibility', 'visible', 'important');
                button.style.setProperty('opacity', '1', 'important');
                button.style.setProperty('pointer-events', 'auto', 'important');
                
                const parent = button.closest('.stButton');
                if (parent) {
                    parent.style.setProperty('display', 'block', 'important');
                    parent.style.setProperty('visibility', 'visible', 'important');
                    parent.style.setProperty('opacity', '1', 'important');
                }
            }
        });
    }
    
    protectNavigationButtons();
    setTimeout(protectNavigationButtons, 50);
    setTimeout(protectNavigationButtons, 100);
    setTimeout(protectNavigationButtons, 200);
    setTimeout(protectNavigationButtons, 500);
    
    const buttonProtectionObserver = new MutationObserver(function(mutations) {
        protectNavigationButtons();
    });
    buttonProtectionObserver.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['style', 'class']
    });
    
    // Protect download buttons from being hidden
    function protectDownloadButtons() {
        const downloadButtons = document.querySelectorAll('.stDownloadButton > button, button[data-testid*="download"]');
        downloadButtons.forEach(function(button) {
            // Force download buttons to always be visible and clickable
            button.style.setProperty('display', 'flex', 'important');
            button.style.setProperty('visibility', 'visible', 'important');
            button.style.setProperty('opacity', '1', 'important');
            button.style.setProperty('pointer-events', 'auto', 'important');
            button.style.setProperty('position', 'relative', 'important');
            
            // Also protect the parent container
            const parent = button.closest('.stDownloadButton');
            if (parent) {
                parent.style.setProperty('display', 'block', 'important');
                parent.style.setProperty('visibility', 'visible', 'important');
                parent.style.setProperty('opacity', '1', 'important');
            }
            
            // Prevent any hover effects that might hide it
            button.onmouseenter = function() {
                this.style.setProperty('display', 'flex', 'important');
                this.style.setProperty('visibility', 'visible', 'important');
                this.style.setProperty('opacity', '1', 'important');
            };
            button.onmouseleave = function() {
                this.style.setProperty('display', 'flex', 'important');
                this.style.setProperty('visibility', 'visible', 'important');
                this.style.setProperty('opacity', '1', 'important');
            };
        });
    }
    
    protectDownloadButtons();
    setTimeout(protectDownloadButtons, 50);
    setTimeout(protectDownloadButtons, 100);
    setTimeout(protectDownloadButtons, 200);
    setTimeout(protectDownloadButtons, 500);
    
    const downloadButtonObserver = new MutationObserver(function(mutations) {
        protectDownloadButtons();
    });
    downloadButtonObserver.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['style', 'class']
    });
    
    // Add visual feedback for button clicks (but skip navigation and download buttons)
    document.addEventListener('click', function(e) {
        const button = e.target.closest('button');
        if (button) {
            // Skip navigation buttons and download buttons to avoid interfering
            const buttonText = (button.textContent || '').trim();
            const buttonKey = button.getAttribute('data-testid') || '';
            const isDownloadButton = button.closest('.stDownloadButton') || buttonKey.includes('download');
            
            if (buttonText.includes('Schema Guide') || 
                buttonText.includes('BigQuery Guide') || 
                buttonText.includes('Back to Main') ||
                buttonText.includes('View BigQuery Guide') ||
                buttonKey.includes('nav_') ||
                isDownloadButton) {
                return; // Don't add animation to navigation or download buttons
            }
            
            if (e.target.closest('.stButton > button')) {
                button.style.transform = 'scale(0.98)';
                setTimeout(function() {
                    button.style.transform = '';
                }, 150);
            }
        }
    });
    
    // Improve focus management for better accessibility
    document.addEventListener('keydown', function(e) {
        // Tab navigation improvements
        if (e.key === 'Tab') {
            document.body.classList.add('keyboard-navigation');
        }
    });
    
    document.addEventListener('mousedown', function() {
        document.body.classList.remove('keyboard-navigation');
    });
    
    // Add loading state animations
    function addLoadingAnimations() {
        const spinners = document.querySelectorAll('[data-testid="stSpinner"]');
        spinners.forEach(function(spinner) {
            spinner.style.opacity = '0';
            setTimeout(function() {
                spinner.style.transition = 'opacity 0.3s ease';
                spinner.style.opacity = '1';
            }, 50);
        });
    }
    
    addLoadingAnimations();
    const loadingObserver = new MutationObserver(function(mutations) {
        addLoadingAnimations();
    });
    loadingObserver.observe(document.body, {
        childList: true,
        subtree: true
    });
    </script>
    """, unsafe_allow_html=True)

def render_shared_logo():
    """Render the shared logo in centered columns."""
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        try:
            st.image("assets/Logo.svg", width=280)
        except:
            st.image("assets/Logo.svg")

def render_shared_footer():
    """Render the shared footer."""
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p style="font-size: 1rem; margin-bottom: 0.25rem;"><strong>BigQuery Data Processor</strong></p>
        <p style="font-size: 0.75rem; color: #9ca3af; margin-bottom: 0.5rem;">Powered by dashworx</p>
        <p style="font-size: 0.9rem; color: #888; margin-bottom: 0.75rem;">Transform your data into BigQuery-ready format | Built with Streamlit</p>
        <p style="font-size: 0.85rem; color: #6b7280; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;">
            <strong>🔒 Privacy & Security:</strong> Your data is processed locally and never stored on our servers. All files are processed in temporary memory and automatically deleted after your session ends.
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_shared_layout():
    """Render all shared layout elements (CSS, logo script, logo, and footer)."""
    render_shared_css()
    render_shared_logo_script()
    render_shared_logo()

# ============================================================================
# MAINTENANCE PAGE
# ============================================================================
def render_maintenance_page():
    """
    Render the maintenance page using shared layout elements.
    This function blocks further execution after rendering.
    """
    # Render shared layout elements
    render_shared_layout()
    
    # Maintenance content
    st.markdown('<h1 class="main-header">BigQuery Data Processor</h1>', unsafe_allow_html=True)
    
    # Maintenance message container with image
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        try:
            st.image("assets/Maintenance.svg", width=200)
        except:
            pass
    
    st.markdown("""
    <div class="maintenance-container">
        <div class="maintenance-title">🔧 Under Maintenance</div>
        <div class="maintenance-message">
            We're currently performing scheduled maintenance to improve your experience.<br>
            Please check back soon. We apologize for any inconvenience.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Render footer
    render_shared_footer()
    
    # Stop execution - nothing from main app should run
    st.stop()

# ============================================================================
# MAIN APP FUNCTIONALITY
# ============================================================================
def create_download_zip(temp_dir):
    """Create a zip file with all output files"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in Path(temp_dir).rglob('*'):
            if file_path.is_file():
                zip_file.write(file_path, file_path.relative_to(temp_dir))
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def create_csv_zip(temp_dir):
    """Create a zip file with only CSV files"""
    zip_buffer = io.BytesIO()
    output_dir = Path(temp_dir)
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        csv_files = list(output_dir.glob("*.csv"))
        for csv_file in csv_files:
            zip_file.write(csv_file, csv_file.name)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def create_schema_zip(temp_dir):
    """Create a zip file with only schema JSON files"""
    zip_buffer = io.BytesIO()
    output_dir = Path(temp_dir)
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        schema_files = list(output_dir.glob("*_bq_schema.json"))
        for schema_file in schema_files:
            zip_file.write(schema_file, schema_file.name)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def create_summary_zip(temp_dir):
    """Create a zip file with only summary TXT files"""
    zip_buffer = io.BytesIO()
    output_dir = Path(temp_dir)
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        summary_files = list(output_dir.glob("*_summary.txt"))
        for summary_file in summary_files:
            zip_file.write(summary_file, summary_file.name)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def create_schema_text_zip(temp_dir):
    """Create a zip file with only schema TXT files"""
    zip_buffer = io.BytesIO()
    output_dir = Path(temp_dir)
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        schema_text_files = list(output_dir.glob("*_bq_schema.txt"))
        for schema_text_file in schema_text_files:
            zip_file.write(schema_text_file, schema_text_file.name)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def perform_initial_inference(df_raw: pd.DataFrame):
    """
    Perform initial type inference on the raw dataframe.
    Returns a dictionary with column info: {col_name: {'type': bq_type, 'sample_values': [...], 'null_count': int}}
    """
    # Clean headers
    df_raw.columns = [simple_header(c) for c in df_raw.columns]
    
    # Clean cells
    df_raw = df_raw.applymap(strip_cell)
    
    schema_info = {}
    
    for col in df_raw.columns:
        # Perform inference
        ser, bq_type, date_fmt = infer_column(df_raw[col], col)
        
        # Get non-null values for sampling
        non_null_series = ser.dropna()
        null_count = len(ser) - len(non_null_series)
        
        # Smart sampling: get 5-7 diverse sample values from different parts
        num_samples = min(7, len(non_null_series))
        sample_values = []
        
        if num_samples > 0:
            if num_samples <= 3:
                # If we have 3 or fewer non-null values, just take all of them
                sample_values = non_null_series.head(num_samples).tolist()
            else:
                # Get diverse samples: first, middle, and last values
                # First value
                sample_values.append(non_null_series.iloc[0])
                
                # Middle values (spread out)
                if num_samples >= 5:
                    mid_start = len(non_null_series) // 4
                    mid_end = 3 * len(non_null_series) // 4
                    middle_indices = [
                        mid_start,
                        len(non_null_series) // 2,
                        mid_end
                    ]
                    for idx in middle_indices:
                        if len(sample_values) < num_samples:
                            sample_values.append(non_null_series.iloc[idx])
                
                # Last value
                if len(sample_values) < num_samples:
                    sample_values.append(non_null_series.iloc[-1])
                
                # Fill remaining slots with evenly spaced values
                if len(sample_values) < num_samples:
                    remaining = num_samples - len(sample_values)
                    step = max(1, len(non_null_series) // (remaining + 1))
                    for i in range(1, remaining + 1):
                        idx = i * step
                        if idx < len(non_null_series) and len(sample_values) < num_samples:
                            if non_null_series.iloc[idx] not in sample_values:
                                sample_values.append(non_null_series.iloc[idx])
                
                # Remove duplicates while preserving order
                seen = set()
                unique_samples = []
                for val in sample_values:
                    if val not in seen:
                        seen.add(val)
                        unique_samples.append(val)
                sample_values = unique_samples[:num_samples]
        
        # Format sample values for display
        sample_display = []
        for val in sample_values:
            if isinstance(val, (int, float)):
                sample_display.append(str(val))
            elif isinstance(val, pd.Timestamp):
                sample_display.append(val.strftime("%Y-%m-%d %H:%M:%S"))
            else:
                str_val = str(val)
                # Truncate long strings
                if len(str_val) > 30:
                    str_val = str_val[:27] + "..."
                sample_display.append(str_val)
        
        schema_info[col] = {
            'type': bq_type,
            'sample_values': sample_display,
            'null_count': null_count,
            'total_count': len(ser)
        }
    
    return schema_info

def display_processing_results(temp_dir):
    """Display processing results in a user-friendly format"""
    
    output_dir = Path(temp_dir)
    if not output_dir.exists():
        st.markdown("""
        <div class="error-box">
            <strong>❌ Error</strong><br>
            No output directory found
        </div>
        """, unsafe_allow_html=True)
        return
    
    csv_files = list(output_dir.glob("*.csv"))
    schema_json_files = list(output_dir.glob("*_bq_schema.json"))
    summary_files = list(output_dir.glob("*_summary.txt"))
    
    st.markdown('<h2>Processing Summary</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <div style="font-size: 2.5rem; font-weight: 700; color: #177091;">{}</div>
            <div style="color: #666; margin-top: 0.5rem;">Processed Sheets</div>
        </div>
        """.format(len(csv_files)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <div style="font-size: 2.5rem; font-weight: 700; color: #177091;">{}</div>
            <div style="color: #666; margin-top: 0.5rem;">Schema (JSON)</div>
        </div>
        """.format(len(schema_json_files)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <div style="font-size: 2.5rem; font-weight: 700; color: #177091;">{}</div>
            <div style="color: #666; margin-top: 0.5rem;">Summary Files</div>
        </div>
        """.format(len(summary_files)), unsafe_allow_html=True)
    
    if not csv_files:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Warning</strong><br>
            No CSV files were generated during processing.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<h2>Processed Files</h2>', unsafe_allow_html=True)
    
    for csv_file in csv_files:
        sheet_name = csv_file.stem
        schema_json_file = output_dir / f"{sheet_name}_bq_schema.json"
        summary_file = output_dir / f"{sheet_name}_summary.txt"
        
        with st.expander(f"Sheet: {sheet_name}", expanded=len(csv_files) == 1):
            
            if schema_json_file.exists():
                with open(schema_json_file, 'r', encoding='utf-8') as f:
                    schema = json.load(f)
                
                st.markdown('<h3>🗂️ BigQuery Schema (JSON)</h3>', unsafe_allow_html=True)
                st.json(schema)
            
            if summary_file.exists():
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary_content = f.read()
                
                st.markdown('<h3>📝 Column Summary</h3>', unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e9ecef; padding: 1rem; border-radius: 8px; font-family: monospace; white-space: pre-wrap; color: #2c3e50;">
                {summary_content}
                </div>
                """, unsafe_allow_html=True)
            
            try:
                df = pd.read_csv(csv_file, nrows=5)
                st.markdown('<h3>📊 Data Preview (First 5 rows)</h3>', unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.markdown(f"""
                <div class="warning-box">
                    <strong>⚠️ Preview Unavailable</strong><br>
                    Could not preview CSV: {str(e)}
                </div>
                """, unsafe_allow_html=True)

def run_main_app():
    """
    Main application functionality.
    This function contains all the actual app logic.
    """
    # Render shared layout elements
    render_shared_layout()
    
    # Header
    st.markdown('<h1 class="main-header">BigQuery Data Processor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Transform your Excel and CSV files into BigQuery-ready format with intelligent data processing</p>', unsafe_allow_html=True)
    
    # Navigation buttons - Centered
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
    with col2:
        if st.button("Schema Guide", use_container_width=True, key="nav_to_schema_docs"):
            try:
                st.switch_page("pages/Schema_Documentation.py")
            except Exception as e:
                st.error(f"Navigation error: {str(e)}")
    with col3:
        if st.button("BigQuery Guide", use_container_width=True, key="nav_to_docs"):
            try:
                st.switch_page("pages/Documentation.py")
            except Exception as e:
                st.error(f"Navigation error: {str(e)}")
    
    # Features section
    st.markdown("""
    <div class="info-text">
        <h3>✨ Key Features</h3>
        <ul>
            <li><strong>Smart Data Type Inference</strong> - Automatically detects STRING, INT64, FLOAT64, BOOL, DATE, and TIMESTAMP types</li>
            <li><strong>Schema Review & Editing</strong> - Review and edit inferred data types before processing with an intuitive interface</li>
            <li><strong>Automatic Column Sanitization</strong> - Converts column names to BigQuery-compatible format</li>
            <li><strong>Data Cleaning & Normalization</strong> - Handles missing values, formats dates, and cleans data</li>
            <li><strong>BigQuery Schema Generation</strong> - Generates ready-to-use JSON format schema files</li>
            <li><strong>Clean CSV Output</strong> - Produces properly formatted CSV files optimized for BigQuery</li>
        </ul>
        <p style="margin-top: 1.25rem; margin-bottom: 0; padding-top: 1.25rem; border-top: 1px solid #e5e7eb;"><strong>💡 Smart Logic:</strong> Columns containing any letters (A-Z) are automatically detected as STRING type. After uploading your file, review the inferred schema, make any adjustments needed, then click "Process with this schema" to generate your BigQuery-ready files!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Important notice section
    st.markdown("""
    <div class="important-notice">
        <h3>⚠️ Important Information</h3>
        <ul>
            <li><strong>Single User Access:</strong> Only one user can access the app at a time</li>
            <li><strong>One File at a Time:</strong> One file can be uploaded at a time. Multiple files cannot be uploaded</li>
            <li><strong>Schema Review:</strong> Choose the data schema correctly before processing the file</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    st.markdown("---")
    st.markdown('<h2 style="display: inline-flex; align-items: center;">Upload Your Data File <span class="upload-alert">Max upload size: 200MB</span></h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an Excel (.xlsx, .xls) or CSV file",
        type=['xlsx', 'xls', 'csv'],
        help="Maximum file size: 200MB",
        label_visibility="collapsed"
    )
    
    # Clear session state if file uploader is cleared (user clicked X button)
    if uploaded_file is None:
        # Check if we had a file before (session state exists)
        if 'uploaded_file_name' in st.session_state:
            # User cleared the file, so clear all session state
            keys_to_clear = [
                'uploaded_file_name', 'schema_review_done', 'inferred_schemas',
                'raw_dataframes', 'processed', 'output_files', 'user_selected_types',
                'sheet_names', 'selected_sheet', 'deleted_columns'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
    
    if uploaded_file is not None:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        
        if file_size_mb > 200:
            st.markdown(f"""
            <div class="error-box">
                <strong>❌ File Size Error</strong><br>
                File size ({file_size_mb:.1f}MB) exceeds the 200MB limit. Please upload a smaller file.
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Check if this is a new file (different from previous upload)
        is_new_file = (
            'uploaded_file_name' not in st.session_state or 
            st.session_state.get('uploaded_file_name') != uploaded_file.name
        )
        
        # Clear all previous session state when a new file is uploaded
        if is_new_file:
            st.session_state['schema_review_done'] = False
            st.session_state['inferred_schemas'] = {}
            st.session_state['raw_dataframes'] = {}
            st.session_state['processed'] = False
            st.session_state['output_files'] = {}
            st.session_state['user_selected_types'] = {}
            st.session_state['sheet_names'] = []
            st.session_state['selected_sheet'] = None
            st.session_state['uploaded_file_name'] = uploaded_file.name
        
        st.markdown(f"""
        <div class="file-info">
            <strong>📄 File:</strong> {uploaded_file.name}<br>
            <strong>📊 Size:</strong> {file_size_mb:.2f} MB<br>
            <strong>✅ Status:</strong> Ready to process
        </div>
        """, unsafe_allow_html=True)
        
        # Perform initial inference if not done yet
        if not st.session_state.get('schema_review_done', False) or st.session_state.get('uploaded_file_name') != uploaded_file.name:
            with st.spinner("Analyzing your file and inferring data types for all sheets..."):
                try:
                    # Load the file
                    file_ext = uploaded_file.name.split('.')[-1].lower()
                    
                    inferred_schemas = {}
                    raw_dataframes = {}
                    sheet_names = []
                    
                    if file_ext in {'xlsx', 'xlsm', 'xls'}:
                        # For Excel, read all sheets
                        uploaded_file.seek(0)
                        all_sheets = pd.read_excel(
                            uploaded_file,
                            sheet_name=None,
                            dtype=str,
                            keep_default_na=False,
                            engine="openpyxl"
                        )
                        
                        # Run inference for each sheet
                        for sheet_name, df_raw in all_sheets.items():
                            sheet_names.append(sheet_name)
                            raw_dataframes[sheet_name] = df_raw
                            schema_info = perform_initial_inference(df_raw.copy())
                            inferred_schemas[sheet_name] = schema_info
                        
                    elif file_ext == 'csv':
                        # For CSV, treat as single sheet
                        uploaded_file.seek(0)
                        df_raw = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False, engine="python", on_bad_lines="skip")
                        sheet_name = uploaded_file.name.split('.')[0]  # Use filename without extension as sheet name
                        sheet_names = [sheet_name]
                        raw_dataframes[sheet_name] = df_raw
                        schema_info = perform_initial_inference(df_raw.copy())
                        inferred_schemas[sheet_name] = schema_info
                    else:
                        st.error("Unsupported file type. Please upload .xlsx, .xls, or .csv files.")
                        return
                    
                    st.session_state['inferred_schemas'] = inferred_schemas
                    st.session_state['raw_dataframes'] = raw_dataframes
                    st.session_state['sheet_names'] = sheet_names
                    # Set first sheet as selected by default
                    if sheet_names:
                        st.session_state['selected_sheet'] = sheet_names[0]
                    st.session_state['uploaded_file_name'] = uploaded_file.name
                    
                except Exception as e:
                    st.markdown(f"""
                    <div class="error-box">
                        <strong>❌ File Analysis Failed</strong><br>
                        {str(e)}
                    </div>
                    """, unsafe_allow_html=True)
                    st.exception(e)
                    return
        
        # Display Schema Review Table
        if st.session_state.get('inferred_schemas') and not st.session_state.get('processed', False):
            st.markdown("---")
            st.markdown('<h2>Schema Review</h2>', unsafe_allow_html=True)
            
            sheet_names = st.session_state.get('sheet_names', [])
            inferred_schemas = st.session_state.get('inferred_schemas', {})
            
            # Initialize user-selected types structure if not exists
            if 'user_selected_types' not in st.session_state:
                st.session_state['user_selected_types'] = {}
            
            # Initialize deleted columns structure if not exists
            if 'deleted_columns' not in st.session_state:
                st.session_state['deleted_columns'] = {}
            
            type_options = ["STRING", "INT64", "FLOAT64", "BOOL", "DATE", "TIMESTAMP"]
            
            # Use tabs for multiple sheets, single view for one sheet
            if len(sheet_names) > 1:
                # Multiple sheets - use tabs
                st.markdown("""
                <div class="info-text" style="margin-bottom: 1.5rem;">
                    <p><span style="color: #dc2626; font-size: 1.2rem; margin-right: 0.5rem;">⚠️</span>This file contains multiple sheets. Each sheet has its own tab below. Review and edit the schema for each sheet separately.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create tabs for each sheet
                tabs = st.tabs(sheet_names)
                
                # Process each sheet in its own tab
                for tab_idx, (tab, sheet_name) in enumerate(zip(tabs, sheet_names)):
                    with tab:
                        # Initialize user-selected types for this sheet if not exists
                        if sheet_name in inferred_schemas:
                            schema_info = inferred_schemas[sheet_name]
                            
                            if sheet_name not in st.session_state['user_selected_types']:
                                st.session_state['user_selected_types'][sheet_name] = {
                                    col: info['type'] for col, info in schema_info.items()
                                }
                            else:
                                # Update user_selected_types for this sheet to match current schema (add new columns, remove old ones)
                                current_types = st.session_state['user_selected_types'][sheet_name]
                                new_types = {}
                                for col, info in schema_info.items():
                                    # Use existing selection if column exists, otherwise use inferred type
                                    new_types[col] = current_types.get(col, info['type'])
                                st.session_state['user_selected_types'][sheet_name] = new_types
                            
                            # Get deleted columns for this sheet
                            deleted_cols = st.session_state.get('deleted_columns', {}).get(sheet_name, set())
                            
                            # Create schema review table for this sheet (exclude deleted columns)
                            review_data = []
                            for col_name, col_info in schema_info.items():
                                # Skip deleted columns
                                if col_name in deleted_cols:
                                    continue
                                    
                                inferred_type = col_info['type']
                                sample_vals = col_info.get('sample_values', [])
                                null_count = col_info.get('null_count', 0)
                                total_count = col_info.get('total_count', 0)
                                
                                # Build sample display with null indicator if needed
                                if sample_vals:
                                    sample_display = ", ".join(sample_vals)
                                    if len(sample_display) > 50:
                                        sample_display = sample_display[:47] + "..."
                                else:
                                    sample_display = "(no data)"
                                
                                # Add null count indicator if there are nulls
                                if null_count > 0:
                                    null_percentage = (null_count / total_count * 100) if total_count > 0 else 0
                                    null_indicator = f" <span style='color: #dc2626; font-size: 0.85rem;'>({null_count} null)</span>"
                                    sample_display = sample_display + null_indicator
                                
                                # Get current user selection for this sheet and column
                                sheet_types = st.session_state['user_selected_types'].get(sheet_name, {})
                                current_selection = sheet_types.get(col_name, inferred_type)
                                
                                review_data.append({
                                    'Column Name': col_name,
                                    'Inferred Type': inferred_type,
                                    'Selected Type': current_selection,
                                    'Sample Values': sample_display
                                })
                            
                            # Display as dataframe for better formatting
                            review_df = pd.DataFrame(review_data)
                            
                            # Info message about form pattern
                            st.info("💡 **Tip:** Make all your schema changes below, then click 'Apply Schema Changes' to update them all at once. This is faster for large datasets!")
                            
                            # Wrap schema review in a form to batch changes
                            with st.form(key=f"schema_form_{sheet_name}", clear_on_submit=False):
                                # Create editable interface using columns with improved styling
                                st.markdown("""
                                <div class="schema-review-container">
                                """, unsafe_allow_html=True)
                                
                                # Header row for column labels
                                header_col1, header_col2, header_col3, header_col4, header_col5 = st.columns([2, 1.5, 1.5, 3, 0.8])
                                with header_col1:
                                    st.markdown('<div style="font-weight: 600; color: #374151; margin-bottom: 0.5rem;">Column Name</div>', unsafe_allow_html=True)
                                with header_col2:
                                    st.markdown('<div style="font-weight: 600; color: #374151; margin-bottom: 0.5rem;">Inferred Type</div>', unsafe_allow_html=True)
                                with header_col3:
                                    st.markdown('<div style="font-weight: 600; color: #374151; margin-bottom: 0.5rem;">Select Type</div>', unsafe_allow_html=True)
                                with header_col4:
                                    st.markdown('<div style="font-weight: 600; color: #374151; margin-bottom: 0.5rem;">Sample Values</div>', unsafe_allow_html=True)
                                with header_col5:
                                    st.markdown('<div style="font-weight: 600; color: #374151; margin-bottom: 0.5rem;">Delete</div>', unsafe_allow_html=True)
                                
                                st.markdown('<hr style="margin: 0.5rem 0 1rem 0; border: none; border-top: 1px solid #e5e7eb;">', unsafe_allow_html=True)
                                
                                # Store form selections temporarily
                                form_selections = {}
                                form_deletions = {}
                                
                                # Display table with editable dropdowns
                                for idx, row in review_df.iterrows():
                                    col_name = row['Column Name']
                                    inferred_type = row['Inferred Type']
                                    sample_vals = row['Sample Values']
                                    
                                    # Add row styling
                                    st.markdown(f'<div class="schema-review-row">', unsafe_allow_html=True)
                                    
                                    col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 3, 0.8])
                                    
                                    with col1:
                                        st.markdown(f'<div class="schema-column-name">{col_name}</div>', unsafe_allow_html=True)
                                    
                                    with col2:
                                        st.markdown(f'<span class="schema-inferred-type">{inferred_type}</span>', unsafe_allow_html=True)
                                    
                                    with col3:
                                        sheet_types = st.session_state['user_selected_types'].get(sheet_name, {})
                                        current_type = sheet_types.get(col_name, inferred_type)
                                        try:
                                            default_index = type_options.index(current_type)
                                        except ValueError:
                                            # If current type is not in options, default to inferred type
                                            default_index = type_options.index(inferred_type) if inferred_type in type_options else 0
                                        
                                        selected_type = st.selectbox(
                                            f"Type for {col_name}",
                                            type_options,
                                            index=default_index,
                                            key=f"form_type_select_{sheet_name}_{col_name}",
                                            label_visibility="collapsed"
                                        )
                                        # Store selection temporarily (won't update session state until form submit)
                                        form_selections[col_name] = selected_type
                                    
                                    with col4:
                                        st.markdown(f'<div class="schema-sample-values">{sample_vals}</div>', unsafe_allow_html=True)
                                    
                                    with col5:
                                        delete_col = st.checkbox(
                                            "Delete",
                                            key=f"form_delete_{sheet_name}_{col_name}",
                                            label_visibility="collapsed",
                                            help=f"Delete column '{col_name}'"
                                        )
                                        form_deletions[col_name] = delete_col
                                    
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    if idx < len(review_df) - 1:
                                        st.markdown('<hr style="margin: 1rem 0; border: none; border-top: 1px solid #e5e7eb;">', unsafe_allow_html=True)
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                                # Check if any columns are marked for deletion by reading checkbox values from session state
                                columns_to_delete = []
                                for col_name in form_deletions.keys():
                                    checkbox_key = f"form_delete_{sheet_name}_{col_name}"
                                    if checkbox_key in st.session_state and st.session_state[checkbox_key]:
                                        columns_to_delete.append(col_name)
                                
                                # Show warning if columns are marked for deletion
                                if columns_to_delete:
                                    st.markdown("""
                                    <div style="background-color: #fef3c7; border-left: 4px solid #f59e0b; padding: 1rem; margin: 1rem 0; border-radius: 4px;">
                                        <strong style="color: #92400e;">⚠️ Warning: Columns Marked for Deletion</strong>
                                        <p style="color: #78350f; margin: 0.5rem 0 0 0;">The following columns will be permanently removed from the processed data:</p>
                                        <ul style="color: #78350f; margin: 0.5rem 0 0 1.5rem;">
                                    """, unsafe_allow_html=True)
                                    for col in columns_to_delete:
                                        st.markdown(f'<li style="color: #78350f;">{col}</li>', unsafe_allow_html=True)
                                    st.markdown("""
                                        </ul>
                                        <p style="color: #78350f; margin: 0.5rem 0 0 0; font-weight: 600;">Please confirm below to proceed with deletion.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Confirmation checkbox
                                    confirm_key = f"confirm_delete_{sheet_name}"
                                    confirm_deletion = st.checkbox(
                                        f"I understand that {len(columns_to_delete)} column(s) will be deleted",
                                        key=confirm_key,
                                        value=st.session_state.get(confirm_key, False)
                                    )
                                else:
                                    confirm_deletion = True  # No deletion, so no confirmation needed
                                
                                # Submit button to apply all changes at once
                                col1, col2, col3 = st.columns([1, 1, 1])
                                with col2:
                                    # Note: We can't use disabled parameter reliably in forms, so we validate on submit instead
                                    apply_changes = st.form_submit_button("Apply Schema Changes", use_container_width=True, type="primary")
                                
                                # Apply all changes when form is submitted
                                # Check confirmation from session state
                                confirm_key = f"confirm_delete_{sheet_name}"
                                is_confirmed = st.session_state.get(confirm_key, True) if columns_to_delete else True
                                if apply_changes:
                                    # Double-check confirmation if columns are marked for deletion
                                    if columns_to_delete and not is_confirmed:
                                        st.error("⚠️ Please confirm that you understand the columns will be deleted by checking the confirmation box.")
                                    elif is_confirmed:
                                        # Update session state with all form selections at once
                                        if sheet_name not in st.session_state['user_selected_types']:
                                            st.session_state['user_selected_types'][sheet_name] = {}
                                        # Batch update all column types
                                        for col_name, selected_type in form_selections.items():
                                            st.session_state['user_selected_types'][sheet_name][col_name] = selected_type
                                        
                                        # Update deleted columns
                                        if sheet_name not in st.session_state['deleted_columns']:
                                            st.session_state['deleted_columns'][sheet_name] = set()
                                        # Add columns marked for deletion
                                        for col_name, should_delete in form_deletions.items():
                                            if should_delete:
                                                st.session_state['deleted_columns'][sheet_name].add(col_name)
                                            else:
                                                # Remove from deleted set if unchecked
                                                st.session_state['deleted_columns'][sheet_name].discard(col_name)
                                        
                                        # Remove deleted columns from user_selected_types
                                        deleted_cols = st.session_state['deleted_columns'][sheet_name]
                                        for deleted_col in deleted_cols:
                                            st.session_state['user_selected_types'][sheet_name].pop(deleted_col, None)
                                        
                                        if columns_to_delete:
                                            st.success(f"✅ Schema changes applied! {len(columns_to_delete)} column(s) deleted. You can now process the file.")
                                        else:
                                            st.success("✅ Schema changes applied! You can now process the file.")
                                        st.rerun()
                            
                            # Process button for this specific sheet (inside the tab)
                            st.markdown("---")
                            process_btn = st.button(
                                f"Process {sheet_name}",
                                type="primary",
                                use_container_width=True,
                                key=f"process_sheet_{sheet_name}"
                            )
                            
                            if process_btn:
                                # Process only this specific sheet
                                with st.spinner(f"Processing {sheet_name}..."):
                                    try:
                                        with tempfile.TemporaryDirectory() as temp_dir:
                                            temp_dir = Path(temp_dir)
                                            
                                            input_path = temp_dir / uploaded_file.name
                                            uploaded_file.seek(0)
                                            with open(input_path, 'wb') as f:
                                                f.write(uploaded_file.getvalue())
                                            
                                            output_dir = temp_dir / "output"
                                            output_dir.mkdir()
                                            
                                            file_ext = input_path.suffix.lower()
                                            
                                            # Get user-selected types for this sheet only
                                            user_selected_types_all = st.session_state.get('user_selected_types', {})
                                            override_types = user_selected_types_all.get(sheet_name, {})
                                            
                                            # Get deleted columns for this sheet
                                            deleted_cols = st.session_state.get('deleted_columns', {}).get(sheet_name, set())
                                            
                                            # Process only this specific sheet
                                            if file_ext in {'.xlsx', '.xlsm', '.xls'}:
                                                # Read only the specific sheet
                                                df_sheet = pd.read_excel(
                                                    input_path,
                                                    sheet_name=sheet_name,
                                                    dtype=str,
                                                    keep_default_na=False,
                                                    engine="openpyxl"
                                                )
                                                # Drop deleted columns before processing
                                                if deleted_cols:
                                                    df_sheet = df_sheet.drop(columns=[col for col in deleted_cols if col in df_sheet.columns], errors='ignore')
                                                process_sheet(sheet_name, df_sheet, output_dir, override_types=override_types)
                                            elif file_ext == '.csv':
                                                # For CSV, process it
                                                df = pd.read_csv(input_path, dtype=str, keep_default_na=False, engine="python", on_bad_lines="skip")
                                                # Drop deleted columns before processing
                                                if deleted_cols:
                                                    df = df.drop(columns=[col for col in deleted_cols if col in df.columns], errors='ignore')
                                                process_sheet(sheet_name, df, output_dir, override_types=override_types)
                                            else:
                                                st.error("Unsupported file type. Please upload .xlsx, .xls, or .csv files.")
                                                return
                                            
                                            # Initialize output_files if not exists
                                            if 'output_files' not in st.session_state:
                                                st.session_state['output_files'] = {}
                                            
                                            # Update only files for this sheet (merge with existing files from other sheets)
                                            for file_path in output_dir.rglob('*'):
                                                if file_path.is_file():
                                                    relative_path = file_path.relative_to(output_dir)
                                                    with open(file_path, 'rb') as f:
                                                        st.session_state['output_files'][str(relative_path)] = f.read()
                                            
                                            # Mark this sheet as processed
                                            if 'processed_sheets' not in st.session_state:
                                                st.session_state['processed_sheets'] = set()
                                            st.session_state['processed_sheets'].add(sheet_name)
                                            
                                            # Check if all sheets are processed
                                            all_sheets = set(st.session_state.get('sheet_names', []))
                                            processed_sheets = st.session_state.get('processed_sheets', set())
                                            if all_sheets.issubset(processed_sheets):
                                                st.session_state['processed'] = True
                                            
                                            st.success(f"✅ {sheet_name} processed successfully!")
                                            st.rerun()
                                    
                                    except Exception as e:
                                        st.markdown(f"""
                                        <div class="error-box">
                                            <strong>❌ Processing Failed for {sheet_name}</strong><br>
                                            {str(e)}
                                        </div>
                                        """, unsafe_allow_html=True)
                                        st.exception(e)
            else:
                # Single sheet - use the only sheet name (no tabs needed)
                selected_sheet = sheet_names[0] if sheet_names else None
                st.session_state['selected_sheet'] = selected_sheet
                st.markdown("""
                <div class="info-text" style="margin-bottom: 1.5rem;">
                    <p>Review and edit the inferred data types below. You can change any column type before processing.</p>
                </div>
                """, unsafe_allow_html=True)
            
            if len(sheet_names) == 1:
                # Single sheet - show schema review (existing code)
                selected_sheet = sheet_names[0] if sheet_names else None
                if selected_sheet and selected_sheet in inferred_schemas:
                    schema_info = inferred_schemas[selected_sheet]
                    
                    # Initialize user-selected types for this sheet if not exists
                    if selected_sheet not in st.session_state['user_selected_types']:
                        st.session_state['user_selected_types'][selected_sheet] = {
                            col: info['type'] for col, info in schema_info.items()
                        }
                    else:
                        # Update user_selected_types for this sheet to match current schema (add new columns, remove old ones)
                        current_types = st.session_state['user_selected_types'][selected_sheet]
                        new_types = {}
                        for col, info in schema_info.items():
                            # Use existing selection if column exists, otherwise use inferred type
                            new_types[col] = current_types.get(col, info['type'])
                        st.session_state['user_selected_types'][selected_sheet] = new_types
                    
                    # Get deleted columns for this sheet
                    deleted_cols = st.session_state.get('deleted_columns', {}).get(selected_sheet, set())
                    
                    # Create schema review table (exclude deleted columns)
                    review_data = []
                    for col_name, col_info in schema_info.items():
                        # Skip deleted columns
                        if col_name in deleted_cols:
                            continue
                            
                        inferred_type = col_info['type']
                        sample_vals = col_info.get('sample_values', [])
                        null_count = col_info.get('null_count', 0)
                        total_count = col_info.get('total_count', 0)
                        
                        # Build sample display with null indicator if needed
                        if sample_vals:
                            sample_display = ", ".join(sample_vals)
                            if len(sample_display) > 50:
                                sample_display = sample_display[:47] + "..."
                        else:
                            sample_display = "(no data)"
                        
                        # Add null count indicator if there are nulls
                        if null_count > 0:
                            null_percentage = (null_count / total_count * 100) if total_count > 0 else 0
                            null_indicator = f" <span style='color: #dc2626; font-size: 0.85rem;'>({null_count} null)</span>"
                            sample_display = sample_display + null_indicator
                        
                        # Get current user selection for this sheet and column
                        sheet_types = st.session_state['user_selected_types'].get(selected_sheet, {})
                        current_selection = sheet_types.get(col_name, inferred_type)
                        
                        review_data.append({
                            'Column Name': col_name,
                            'Inferred Type': inferred_type,
                            'Selected Type': current_selection,
                            'Sample Values': sample_display
                        })
                    
                    # Display as dataframe for better formatting
                    review_df = pd.DataFrame(review_data)
                    
                    # Info message about form pattern
                    st.info("💡 **Tip:** Make all your schema changes below, then click 'Apply Schema Changes' to update them all at once. This is faster for large datasets!")
                    
                    # Wrap schema review in a form to batch changes
                    with st.form(key=f"schema_form_{selected_sheet}", clear_on_submit=False):
                        # Create editable interface using columns with improved styling
                        st.markdown("""
                        <div class="schema-review-container">
                        """, unsafe_allow_html=True)
                        
                        # Header row for column labels
                        header_col1, header_col2, header_col3, header_col4, header_col5 = st.columns([2, 1.5, 1.5, 3, 0.8])
                        with header_col1:
                            st.markdown('<div style="font-weight: 600; color: #374151; margin-bottom: 0.5rem;">Column Name</div>', unsafe_allow_html=True)
                        with header_col2:
                            st.markdown('<div style="font-weight: 600; color: #374151; margin-bottom: 0.5rem;">Inferred Type</div>', unsafe_allow_html=True)
                        with header_col3:
                            st.markdown('<div style="font-weight: 600; color: #374151; margin-bottom: 0.5rem;">Select Type</div>', unsafe_allow_html=True)
                        with header_col4:
                            st.markdown('<div style="font-weight: 600; color: #374151; margin-bottom: 0.5rem;">Sample Values</div>', unsafe_allow_html=True)
                        with header_col5:
                            st.markdown('<div style="font-weight: 600; color: #374151; margin-bottom: 0.5rem;">Delete</div>', unsafe_allow_html=True)
                        
                        st.markdown('<hr style="margin: 0.5rem 0 1rem 0; border: none; border-top: 1px solid #e5e7eb;">', unsafe_allow_html=True)
                        
                        # Store form selections temporarily
                        form_selections = {}
                        form_deletions = {}
                        
                        # Display table with editable dropdowns
                        for idx, row in review_df.iterrows():
                            col_name = row['Column Name']
                            inferred_type = row['Inferred Type']
                            sample_vals = row['Sample Values']
                            
                            # Add row styling
                            st.markdown(f'<div class="schema-review-row">', unsafe_allow_html=True)
                            
                            col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 3, 0.8])
                            
                            with col1:
                                st.markdown(f'<div class="schema-column-name">{col_name}</div>', unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f'<span class="schema-inferred-type">{inferred_type}</span>', unsafe_allow_html=True)
                            
                            with col3:
                                sheet_types = st.session_state['user_selected_types'].get(selected_sheet, {})
                                current_type = sheet_types.get(col_name, inferred_type)
                                try:
                                    default_index = type_options.index(current_type)
                                except ValueError:
                                    # If current type is not in options, default to inferred type
                                    default_index = type_options.index(inferred_type) if inferred_type in type_options else 0
                                
                                selected_type = st.selectbox(
                                    f"Type for {col_name}",
                                    type_options,
                                    index=default_index,
                                    key=f"form_type_select_{selected_sheet}_{col_name}",
                                    label_visibility="collapsed"
                                )
                                # Store selection temporarily (won't update session state until form submit)
                                form_selections[col_name] = selected_type
                            
                            with col4:
                                st.markdown(f'<div class="schema-sample-values">{sample_vals}</div>', unsafe_allow_html=True)
                            
                            with col5:
                                delete_col = st.checkbox(
                                    "Delete",
                                    key=f"form_delete_{selected_sheet}_{col_name}",
                                    label_visibility="collapsed",
                                    help=f"Delete column '{col_name}'"
                                )
                                form_deletions[col_name] = delete_col
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            if idx < len(review_df) - 1:
                                st.markdown('<hr style="margin: 1rem 0; border: none; border-top: 1px solid #e5e7eb;">', unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Check if any columns are marked for deletion by reading checkbox values from session state
                        columns_to_delete = []
                        for col_name in form_deletions.keys():
                            checkbox_key = f"form_delete_{selected_sheet}_{col_name}"
                            if checkbox_key in st.session_state and st.session_state[checkbox_key]:
                                columns_to_delete.append(col_name)
                        
                        # Show warning if columns are marked for deletion
                        if columns_to_delete:
                            st.markdown("""
                            <div style="background-color: #fef3c7; border-left: 4px solid #f59e0b; padding: 1rem; margin: 1rem 0; border-radius: 4px;">
                                <strong style="color: #92400e;">⚠️ Warning: Columns Marked for Deletion</strong>
                                <p style="color: #78350f; margin: 0.5rem 0 0 0;">The following columns will be permanently removed from the processed data:</p>
                                <ul style="color: #78350f; margin: 0.5rem 0 0 1.5rem;">
                            """, unsafe_allow_html=True)
                            for col in columns_to_delete:
                                st.markdown(f'<li style="color: #78350f;">{col}</li>', unsafe_allow_html=True)
                            st.markdown("""
                                </ul>
                                <p style="color: #78350f; margin: 0.5rem 0 0 0; font-weight: 600;">Please confirm below to proceed with deletion.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Confirmation checkbox
                            confirm_key = f"confirm_delete_{selected_sheet}"
                            confirm_deletion = st.checkbox(
                                f"I understand that {len(columns_to_delete)} column(s) will be deleted",
                                key=confirm_key,
                                value=st.session_state.get(confirm_key, False)
                            )
                        else:
                            confirm_deletion = True  # No deletion, so no confirmation needed
                        
                        # Submit button to apply all changes at once
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col2:
                            # Note: We can't use disabled parameter reliably in forms, so we validate on submit instead
                            apply_changes = st.form_submit_button("Apply Schema Changes", use_container_width=True, type="primary")
                        
                        # Apply all changes when form is submitted
                        # Check confirmation from session state
                        confirm_key = f"confirm_delete_{selected_sheet}"
                        is_confirmed = st.session_state.get(confirm_key, True) if columns_to_delete else True
                        if apply_changes:
                            # Double-check confirmation if columns are marked for deletion
                            if columns_to_delete and not is_confirmed:
                                st.error("⚠️ Please confirm that you understand the columns will be deleted by checking the confirmation box.")
                            elif is_confirmed:
                                # Update session state with all form selections at once
                                if selected_sheet not in st.session_state['user_selected_types']:
                                    st.session_state['user_selected_types'][selected_sheet] = {}
                                # Batch update all column types
                                for col_name, selected_type in form_selections.items():
                                    st.session_state['user_selected_types'][selected_sheet][col_name] = selected_type
                                
                                # Update deleted columns
                                if selected_sheet not in st.session_state['deleted_columns']:
                                    st.session_state['deleted_columns'][selected_sheet] = set()
                                # Add columns marked for deletion
                                for col_name, should_delete in form_deletions.items():
                                    if should_delete:
                                        st.session_state['deleted_columns'][selected_sheet].add(col_name)
                                    else:
                                        # Remove from deleted set if unchecked
                                        st.session_state['deleted_columns'][selected_sheet].discard(col_name)
                                
                                # Remove deleted columns from user_selected_types
                                deleted_cols = st.session_state['deleted_columns'][selected_sheet]
                                for deleted_col in deleted_cols:
                                    st.session_state['user_selected_types'][selected_sheet].pop(deleted_col, None)
                                
                                if columns_to_delete:
                                    st.success(f"✅ Schema changes applied! {len(columns_to_delete)} column(s) deleted. You can now process the file.")
                                else:
                                    st.success("✅ Schema changes applied! You can now process the file.")
                                st.rerun()
                    
                    # Process button for single sheet (outside form, after schema review)
                    st.markdown("---")
                    process_with_schema_btn = st.button("Process with this schema", type="primary", use_container_width=True, key="process_with_schema")
                    
                    if process_with_schema_btn:
                        with st.spinner("Processing your file with the selected schemas..."):
                            try:
                                with tempfile.TemporaryDirectory() as temp_dir:
                                    temp_dir = Path(temp_dir)
                                    
                                    input_path = temp_dir / uploaded_file.name
                                    # Reset file pointer
                                    uploaded_file.seek(0)
                                    with open(input_path, 'wb') as f:
                                        f.write(uploaded_file.getvalue())
                                    
                                    output_dir = temp_dir / "output"
                                    output_dir.mkdir()
                                    
                                    file_ext = input_path.suffix.lower()
                                    
                                    # Get user-selected types per sheet
                                    user_selected_types_all = st.session_state.get('user_selected_types', {})
                                    
                                    # Process with override_types per sheet
                                    if file_ext in {'.xlsx', '.xlsm', '.xls'}:
                                        # For Excel, process all sheets with their respective override_types
                                        sheets = pd.read_excel(
                                            input_path,
                                            sheet_name=None,
                                            dtype=str,
                                            keep_default_na=False,
                                            engine="openpyxl"
                                        )
                                        for sheet_name, df_sheet in sheets.items():
                                            # Get override_types for this specific sheet
                                            override_types = user_selected_types_all.get(sheet_name, {})
                                            # Get deleted columns for this sheet
                                            deleted_cols = st.session_state.get('deleted_columns', {}).get(sheet_name, set())
                                            # Drop deleted columns before processing
                                            if deleted_cols:
                                                df_sheet = df_sheet.drop(columns=[col for col in deleted_cols if col in df_sheet.columns], errors='ignore')
                                            process_sheet(sheet_name, df_sheet, output_dir, override_types=override_types)
                                    elif file_ext == '.csv':
                                        # For CSV, use the sheet name (filename without extension)
                                        sheet_name = input_path.stem
                                        df = pd.read_csv(input_path, dtype=str, keep_default_na=False, engine="python", on_bad_lines="skip")
                                        override_types = user_selected_types_all.get(sheet_name, {})
                                        # Get deleted columns for this sheet
                                        deleted_cols = st.session_state.get('deleted_columns', {}).get(sheet_name, set())
                                        # Drop deleted columns before processing
                                        if deleted_cols:
                                            df = df.drop(columns=[col for col in deleted_cols if col in df.columns], errors='ignore')
                                        process_sheet(sheet_name, df, output_dir, override_types=override_types)
                                    else:
                                        st.error("Unsupported file type. Please upload .xlsx, .xls, or .csv files.")
                                        return
                                    
                                    # Store output files
                                    st.session_state['output_files'] = {}
                                    for file_path in output_dir.rglob('*'):
                                        if file_path.is_file():
                                            relative_path = file_path.relative_to(output_dir)
                                            with open(file_path, 'rb') as f:
                                                st.session_state['output_files'][str(relative_path)] = f.read()
                                    
                                    st.session_state['processed'] = True
                                    st.session_state['schema_review_done'] = True
                                    st.rerun()
                            
                            except Exception as e:
                                st.markdown(f"""
                                <div class="error-box">
                                    <strong>❌ Processing Failed</strong><br>
                                    {str(e)}
                                </div>
                                """, unsafe_allow_html=True)
                                st.exception(e)
                                st.session_state['processed'] = False
                        
        # Only show processed results if we have a file and it's been processed (or has output files)
        # For multi-sheet: show if any sheets are processed (has output_files)
        # For single sheet: show if processed flag is True
        has_output_files = bool(st.session_state.get('output_files', {}))
        is_processed = st.session_state.get('processed', False)
        should_show_downloads = (uploaded_file is not None and 
                                 st.session_state.get('uploaded_file_name') == uploaded_file.name and
                                 (is_processed or has_output_files))
        
        if should_show_downloads:
            with tempfile.TemporaryDirectory() as display_temp_dir:
                display_temp_dir = Path(display_temp_dir)
                output_dir = display_temp_dir / "output"
                output_dir.mkdir()
                
                for file_path_str, file_content in st.session_state.get('output_files', {}).items():
                    file_path = output_dir / file_path_str
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(file_path, 'wb') as f:
                        f.write(file_content)
                
                st.markdown("""
                <div class="success-box">
                    <strong>✅ Processing Completed Successfully!</strong><br>
                    Your data has been processed and is now BigQuery-ready.
                </div>
                """, unsafe_allow_html=True)
                
                display_processing_results(output_dir)
                
                st.markdown("---")
                st.markdown('<h2>Download Results</h2>', unsafe_allow_html=True)
                
                base_name = st.session_state.get('uploaded_file_name', 'processed').split('.')[0]
                
                # Get all files from output directory
                output_files_list = []
                for file_path_str in st.session_state.get('output_files', {}).keys():
                    output_files_list.append(Path(file_path_str))
                
                # Group files by sheet name
                # Extract sheet name from filename (e.g., "Sheet1.csv" -> "Sheet1")
                files_by_sheet = {}
                for file_path in output_files_list:
                    # Remove extension and any suffix like "_bq_schema" or "_summary"
                    base_name_file = file_path.stem
                    # Remove known suffixes
                    for suffix in ['_bq_schema', '_summary']:
                        if base_name_file.endswith(suffix):
                            base_name_file = base_name_file[:-len(suffix)]
                    
                    if base_name_file not in files_by_sheet:
                        files_by_sheet[base_name_file] = []
                    files_by_sheet[base_name_file].append(file_path)
                
                # Sort sheets and files within each sheet
                for sheet_name in files_by_sheet:
                    files_by_sheet[sheet_name].sort(key=lambda x: x.name)
                
                # Get sheet names from session state
                sheet_names = st.session_state.get('sheet_names', [])
                
                # Filter to only show tabs for sheets that have been processed (have files)
                processed_sheet_names = [name for name in sheet_names if name in files_by_sheet]
                
                # Use tabs for multiple sheets, single view for one sheet
                if len(processed_sheet_names) > 1:
                    # Multiple processed sheets - use tabs for downloads (only show processed sheets)
                    download_tabs = st.tabs(processed_sheet_names)
                    
                    # Process each sheet in its own download tab
                    for tab_idx, (tab, sheet_name) in enumerate(zip(download_tabs, processed_sheet_names)):
                        with tab:
                            if sheet_name in files_by_sheet:
                                sheet_files = files_by_sheet[sheet_name]
                                
                                st.markdown('<h3 style="margin-top: 0;">Individual File Downloads</h3>', unsafe_allow_html=True)
                                
                                # Group files by type for this sheet
                                csv_files = [f for f in sheet_files if f.suffix.lower() == '.csv']
                                schema_json_files = [f for f in sheet_files if '_bq_schema.json' in f.name]
                                summary_files = [f for f in sheet_files if '_summary.txt' in f.name]
                                
                                # Display CSV file
                                if csv_files:
                                    num_cols = min(4, len(csv_files))
                                    cols = st.columns(num_cols, gap="small")
                                    for idx, file_path in enumerate(csv_files):
                                        relative_path_str = str(file_path)
                                        file_data = st.session_state.get('output_files', {}).get(relative_path_str, b'')
                                        if file_data:
                                            with cols[idx % len(cols)]:
                                                display_name = "Cleaned CSV" if len(csv_files) == 1 else file_path.name
                                                if len(display_name) > 25:
                                                    display_name = display_name[:22] + "..."
                                                st.download_button(
                                                    label=display_name,
                                                    data=file_data,
                                                    file_name=file_path.name,
                                                    mime="text/csv",
                                                    key=f"csv_{sheet_name}_{file_path.name}_{idx}",
                                                    use_container_width=True
                                                )
                                
                                # Display Schema JSON file
                                if schema_json_files:
                                    num_cols = min(4, len(schema_json_files))
                                    cols = st.columns(num_cols, gap="small")
                                    for idx, file_path in enumerate(schema_json_files):
                                        relative_path_str = str(file_path)
                                        file_data = st.session_state.get('output_files', {}).get(relative_path_str, b'')
                                        if file_data:
                                            with cols[idx % len(cols)]:
                                                display_name = "Schema (JSON)" if len(schema_json_files) == 1 else file_path.name
                                                if len(display_name) > 25:
                                                    display_name = display_name[:22] + "..."
                                                st.download_button(
                                                    label=display_name,
                                                    data=file_data,
                                                    file_name=file_path.name,
                                                    mime="application/json",
                                                    key=f"schema_json_{sheet_name}_{file_path.name}_{idx}",
                                                    use_container_width=True
                                                )
                                
                                # Display Summary file
                                if summary_files:
                                    num_cols = min(4, len(summary_files))
                                    cols = st.columns(num_cols, gap="small")
                                    for idx, file_path in enumerate(summary_files):
                                        relative_path_str = str(file_path)
                                        file_data = st.session_state.get('output_files', {}).get(relative_path_str, b'')
                                        if file_data:
                                            with cols[idx % len(cols)]:
                                                display_name = "Summary" if len(summary_files) == 1 else file_path.name
                                                if len(display_name) > 25:
                                                    display_name = display_name[:22] + "..."
                                                st.download_button(
                                                    label=display_name,
                                                    data=file_data,
                                                    file_name=file_path.name,
                                                    mime="text/plain",
                                                    key=f"summary_{sheet_name}_{file_path.name}_{idx}",
                                                    use_container_width=True
                                                )
                            else:
                                st.info(f"No files found for sheet: {sheet_name}")
                    
                    # Bulk download buttons (for all processed sheets) - shown after tabs
                    if len(processed_sheet_names) > 0:
                        st.markdown("---")
                        st.markdown('<h3>Bulk Downloads (All Processed Sheets)</h3>', unsafe_allow_html=True)
                    
                    with tempfile.TemporaryDirectory() as zip_temp_dir:
                        zip_temp_dir = Path(zip_temp_dir)
                        zip_output_dir = zip_temp_dir / "output"
                        zip_output_dir.mkdir()
                        
                        for file_path_str, file_content in st.session_state.get('output_files', {}).items():
                            file_path = zip_output_dir / file_path_str
                            file_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(file_path, 'wb') as f:
                                f.write(file_content)
                        
                        csv_zip_data = create_csv_zip(zip_output_dir)
                        schema_json_zip_data = create_schema_zip(zip_output_dir)
                        summary_zip_data = create_summary_zip(zip_output_dir)
                        all_zip_data = create_download_zip(zip_output_dir)
                    
                    # Center the three buttons with spacer columns
                    col_spacer1, col1, col2, col3, col_spacer2 = st.columns([1, 1.5, 1.5, 1.5, 1])
                    
                    with col1:
                        st.download_button(
                            label="Download CSV Files",
                            data=csv_zip_data,
                            file_name=f"{base_name}_csv_files.zip",
                            mime="application/zip",
                            help="Download all processed CSV files",
                            use_container_width=True,
                            key=f"bulk_csv_{base_name}"
                        )
                    
                    with col2:
                        st.download_button(
                            label="Download Schema (JSON)",
                            data=schema_json_zip_data,
                            file_name=f"{base_name}_schemas_json.zip",
                            mime="application/zip",
                            help="Download all BigQuery schema JSON files",
                            use_container_width=True,
                            key=f"bulk_schema_json_{base_name}"
                        )
                    
                    with col3:
                        st.download_button(
                            label="Download All (ZIP)",
                            data=all_zip_data,
                            file_name=f"{base_name}_all_files.zip",
                            mime="application/zip",
                            help="Download all processed files including CSV, JSON schema, and summary files",
                            use_container_width=True,
                            key=f"bulk_all_{base_name}"
                        )
                elif len(processed_sheet_names) == 1:
                    # Single processed sheet - show downloads in single view (no tabs)
                    sheet_name = processed_sheet_names[0]
                    sheet_files = files_by_sheet[sheet_name]
                    
                    st.markdown('<h3>Individual File Downloads</h3>', unsafe_allow_html=True)
                    
                    # Group files by type for this sheet
                    csv_files = [f for f in sheet_files if f.suffix.lower() == '.csv']
                    schema_json_files = [f for f in sheet_files if '_bq_schema.json' in f.name]
                    summary_files = [f for f in sheet_files if '_summary.txt' in f.name]
                    
                    # Display CSV file
                    if csv_files:
                        num_cols = min(4, len(csv_files))
                        cols = st.columns(num_cols, gap="small")
                        for idx, file_path in enumerate(csv_files):
                            relative_path_str = str(file_path)
                            file_data = st.session_state.get('output_files', {}).get(relative_path_str, b'')
                            if file_data:
                                with cols[idx % len(cols)]:
                                    display_name = "Cleaned CSV" if len(csv_files) == 1 else file_path.name
                                    if len(display_name) > 25:
                                        display_name = display_name[:22] + "..."
                                    st.download_button(
                                        label=display_name,
                                        data=file_data,
                                        file_name=file_path.name,
                                        mime="text/csv",
                                        key=f"csv_{sheet_name}_{file_path.name}_{idx}",
                                        use_container_width=True
                                    )
                    
                    # Display Schema JSON file
                    if schema_json_files:
                        num_cols = min(4, len(schema_json_files))
                        cols = st.columns(num_cols, gap="small")
                        for idx, file_path in enumerate(schema_json_files):
                            relative_path_str = str(file_path)
                            file_data = st.session_state.get('output_files', {}).get(relative_path_str, b'')
                            if file_data:
                                with cols[idx % len(cols)]:
                                    display_name = "Schema (JSON)" if len(schema_json_files) == 1 else file_path.name
                                    if len(display_name) > 25:
                                        display_name = display_name[:22] + "..."
                                    st.download_button(
                                        label=display_name,
                                        data=file_data,
                                        file_name=file_path.name,
                                        mime="application/json",
                                        key=f"schema_json_{sheet_name}_{file_path.name}_{idx}",
                                        use_container_width=True
                                    )
                    
                    # Display Summary file
                    if summary_files:
                        num_cols = min(4, len(summary_files))
                        cols = st.columns(num_cols, gap="small")
                        for idx, file_path in enumerate(summary_files):
                            relative_path_str = str(file_path)
                            file_data = st.session_state.get('output_files', {}).get(relative_path_str, b'')
                            if file_data:
                                with cols[idx % len(cols)]:
                                    display_name = "Summary" if len(summary_files) == 1 else file_path.name
                                    if len(display_name) > 25:
                                        display_name = display_name[:22] + "..."
                                    st.download_button(
                                        label=display_name,
                                        data=file_data,
                                        file_name=file_path.name,
                                        mime="text/plain",
                                        key=f"summary_{sheet_name}_{file_path.name}_{idx}",
                                        use_container_width=True
                                    )
                    
                    # Bulk download buttons
                    st.markdown("---")
                    st.markdown('<h3>Bulk Downloads</h3>', unsafe_allow_html=True)
                    
                    with tempfile.TemporaryDirectory() as zip_temp_dir:
                        zip_temp_dir = Path(zip_temp_dir)
                        zip_output_dir = zip_temp_dir / "output"
                        zip_output_dir.mkdir()
                        
                        for file_path_str, file_content in st.session_state.get('output_files', {}).items():
                            file_path = zip_output_dir / file_path_str
                            file_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(file_path, 'wb') as f:
                                f.write(file_content)
                        
                        csv_zip_data = create_csv_zip(zip_output_dir)
                        schema_json_zip_data = create_schema_zip(zip_output_dir)
                        summary_zip_data = create_summary_zip(zip_output_dir)
                        all_zip_data = create_download_zip(zip_output_dir)
                    
                    # Center the three buttons with spacer columns
                    col_spacer1, col1, col2, col3, col_spacer2 = st.columns([1, 1.5, 1.5, 1.5, 1])
                    
                    with col1:
                        st.download_button(
                            label="Download CSV Files",
                            data=csv_zip_data,
                            file_name=f"{base_name}_csv_files.zip",
                            mime="application/zip",
                            help="Download all processed CSV files",
                            use_container_width=True,
                            key=f"bulk_csv_{base_name}"
                        )
                    
                    with col2:
                        st.download_button(
                            label="Download Schema (JSON)",
                            data=schema_json_zip_data,
                            file_name=f"{base_name}_schemas_json.zip",
                            mime="application/zip",
                            help="Download all BigQuery schema JSON files",
                            use_container_width=True,
                            key=f"bulk_schema_json_{base_name}"
                        )
                    
                    with col3:
                        st.download_button(
                            label="Download All (ZIP)",
                            data=all_zip_data,
                            file_name=f"{base_name}_all_files.zip",
                            mime="application/zip",
                            help="Download all processed files including CSV, JSON schema, and summary files",
                            use_container_width=True,
                            key=f"bulk_all_{base_name}"
                        )
                else:
                    # No processed sheets or single sheet file - show downloads in single view (existing behavior)
                    with tempfile.TemporaryDirectory() as zip_temp_dir:
                        zip_temp_dir = Path(zip_temp_dir)
                        zip_output_dir = zip_temp_dir / "output"
                        zip_output_dir.mkdir()
                        
                        for file_path_str, file_content in st.session_state.get('output_files', {}).items():
                            file_path = zip_output_dir / file_path_str
                            file_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(file_path, 'wb') as f:
                                f.write(file_content)
                        
                        csv_zip_data = create_csv_zip(zip_output_dir)
                        schema_json_zip_data = create_schema_zip(zip_output_dir)
                        summary_zip_data = create_summary_zip(zip_output_dir)
                        all_zip_data = create_download_zip(zip_output_dir)
                    
                    # Individual file downloads section
                    st.markdown('<h3>Individual File Downloads</h3>', unsafe_allow_html=True)
                    
                    # Display files for the single sheet
                    if files_by_sheet:
                        sheet_name = list(files_by_sheet.keys())[0]
                        sheet_files = files_by_sheet[sheet_name]
                        
                        # Group files by type for this sheet
                        csv_files = [f for f in sheet_files if f.suffix.lower() == '.csv']
                        schema_json_files = [f for f in sheet_files if '_bq_schema.json' in f.name]
                        summary_files = [f for f in sheet_files if '_summary.txt' in f.name]
                        
                        # Display CSV file
                        if csv_files:
                            num_cols = min(4, len(csv_files))
                            cols = st.columns(num_cols, gap="small")
                            for idx, file_path in enumerate(csv_files):
                                relative_path_str = str(file_path)
                                file_data = st.session_state.get('output_files', {}).get(relative_path_str, b'')
                                if file_data:
                                    with cols[idx % len(cols)]:
                                        display_name = "Cleaned CSV" if len(csv_files) == 1 else file_path.name
                                        if len(display_name) > 25:
                                            display_name = display_name[:22] + "..."
                                st.download_button(
                                            label=display_name,
                                    data=file_data,
                                    file_name=file_path.name,
                                            mime="text/csv",
                                            key=f"csv_{sheet_name}_{file_path.name}_{idx}",
                                            use_container_width=True
                                        )
                        
                        # Display Schema JSON file
                        if schema_json_files:
                            num_cols = min(4, len(schema_json_files))
                            cols = st.columns(num_cols, gap="small")
                            for idx, file_path in enumerate(schema_json_files):
                                relative_path_str = str(file_path)
                                file_data = st.session_state.get('output_files', {}).get(relative_path_str, b'')
                                if file_data:
                                    with cols[idx % len(cols)]:
                                        display_name = "Schema (JSON)" if len(schema_json_files) == 1 else file_path.name
                                        if len(display_name) > 25:
                                            display_name = display_name[:22] + "..."
                                        st.download_button(
                                            label=display_name,
                                            data=file_data,
                                            file_name=file_path.name,
                                            mime="application/json",
                                            key=f"schema_json_{sheet_name}_{file_path.name}_{idx}",
                                            use_container_width=True
                                        )
                        
                        # Display Summary file
                        if summary_files:
                            num_cols = min(4, len(summary_files))
                            cols = st.columns(num_cols, gap="small")
                            for idx, file_path in enumerate(summary_files):
                                relative_path_str = str(file_path)
                                file_data = st.session_state.get('output_files', {}).get(relative_path_str, b'')
                                if file_data:
                                    with cols[idx % len(cols)]:
                                        display_name = "Summary" if len(summary_files) == 1 else file_path.name
                                        if len(display_name) > 25:
                                            display_name = display_name[:22] + "..."
                                        st.download_button(
                                            label=display_name,
                                            data=file_data,
                                            file_name=file_path.name,
                                            mime="text/plain",
                                            key=f"summary_{sheet_name}_{file_path.name}_{idx}",
                                            use_container_width=True
                                        )
                    
                    # Bulk download buttons
                    st.markdown("---")
                    st.markdown('<h3>Bulk Downloads</h3>', unsafe_allow_html=True)
                    # Center the three buttons with spacer columns
                    col_spacer1, col1, col2, col3, col_spacer2 = st.columns([1, 1.5, 1.5, 1.5, 1])
                    
                    with col1:
                        st.download_button(
                            label="Download CSV Files",
                            data=csv_zip_data,
                            file_name=f"{base_name}_csv_files.zip",
                            mime="application/zip",
                            help="Download all processed CSV files",
                            use_container_width=True,
                            key=f"bulk_csv_{base_name}"
                        )
                    
                    with col2:
                        st.download_button(
                            label="Download Schema (JSON)",
                            data=schema_json_zip_data,
                            file_name=f"{base_name}_schemas_json.zip",
                            mime="application/zip",
                            help="Download all BigQuery schema JSON files",
                            use_container_width=True,
                            key=f"bulk_schema_json_{base_name}"
                        )
                    
                    with col3:
                        st.download_button(
                            label="Download All (ZIP)",
                            data=all_zip_data,
                            file_name=f"{base_name}_all_files.zip",
                            mime="application/zip",
                            help="Download all processed files including CSV, JSON schema, and summary files",
                            use_container_width=True,
                            key=f"bulk_all_{base_name}"
                        )
    
    # Footer
    render_shared_footer()

# ============================================================================
# MAIN ENTRY POINT - CONDITIONAL SWITCH
# ============================================================================
if __name__ == "__main__":
    # Check maintenance mode first - this is the critical switch
    if is_maintenance_mode():
        render_maintenance_page()
        # render_maintenance_page() calls st.stop(), so nothing below runs
    else:
        run_main_app()
