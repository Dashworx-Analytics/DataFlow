import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Schema Documentation - BigQuery Data Processor",
    page_icon="assets/web/icons8-hub-pulsar-gradient-32.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Same styling as main page
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
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
    
    body {
        background-color: #f8f9fa;
    }
    
    .main {
        background-color: #f8f9fa;
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
    
    /* Button styling */
    .stButton > button {
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
    
    .stButton > button:hover {
        background: #335169 !important;
        background-color: #335169 !important;
        color: white !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* Section headers */
    h2 {
        color: #111827;
        font-weight: 600;
        margin-top: 2.5rem;
        margin-bottom: 1.25rem;
        font-size: 1.5rem;
        letter-spacing: -0.01em;
    }
    
    h3 {
        color: #111827;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-size: 1.25rem;
        letter-spacing: -0.01em;
    }
    
    /* Info card styling */
    .info-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
        line-height: 1.7;
        color: #374151;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .info-card h3 {
        color: #111827 !important;
        margin-top: 0 !important;
        margin-bottom: 1.25rem !important;
        font-size: 1.25rem;
        font-weight: 600;
    }
    
    .info-card ul {
        color: #374151;
        margin: 0 0 1rem 0;
        padding-left: 1.5rem;
    }
    
    .info-card ul li {
        margin-bottom: 0.75rem;
        line-height: 1.6;
    }
    
    .info-card p {
        color: #374151;
        margin: 0.75rem 0;
    }
    
    /* Important notice styling */
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
    
    .important-notice p {
        color: #7f1d1d;
        margin: 0.75rem 0;
    }
    
    .info-card strong {
        color: #111827;
        font-weight: 600;
    }
    
    /* Example box styling */
    .example-box {
        background: #f8f9fa;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #177091;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        color: #2c3e50;
        white-space: pre-wrap;
        overflow-x: auto;
    }
    
    .example-box-title {
        font-weight: 600;
        color: #177091;
        margin-bottom: 0.5rem;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Type badge styling */
    .type-badge {
        display: inline-block;
        background: #177091;
        color: #ffffff;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.875rem;
        margin-right: 0.5rem;
        font-family: 'Courier New', monospace;
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
    
    .main {
        color: #374151;
    }
    
    .stMarkdown {
        color: #374151;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Add JavaScript to center logo and remove image enlarge buttons
    st.markdown("""
    <script>
    function centerLogo() {
        // Find the middle column (2nd column)
        const middleColumn = document.querySelector('div[data-testid="column"]:nth-of-type(2)');
        if (middleColumn) {
            // Force center alignment
            middleColumn.style.display = 'flex';
            middleColumn.style.justifyContent = 'center';
            middleColumn.style.alignItems = 'center';
            middleColumn.style.textAlign = 'center';
            middleColumn.style.width = '100%';
            middleColumn.style.margin = '0 auto';
            
            // Find image in middle column and center it
            const logoImage = middleColumn.querySelector('[data-testid="stImage"]');
            if (logoImage) {
                logoImage.style.margin = '0 auto';
                logoImage.style.display = 'block';
                logoImage.style.textAlign = 'center';
                
                // Also center any nested divs or images
                const nestedDivs = logoImage.querySelectorAll('div, img');
                nestedDivs.forEach(function(el) {
                    el.style.margin = '0 auto';
                    el.style.display = 'block';
                    el.style.textAlign = 'center';
                });
            }
        }
    }
    
    // Run immediately and on delays
    centerLogo();
    setTimeout(centerLogo, 50);
    setTimeout(centerLogo, 100);
    setTimeout(centerLogo, 200);
    setTimeout(centerLogo, 500);
    
    // Watch for DOM changes
    const centerObserver = new MutationObserver(centerLogo);
    centerObserver.observe(document.body, { childList: true, subtree: true });
    
    function removeLogoButtons() {
        // Only target logo images - check if image is in the middle column (logo position)
        const middleColumn = document.querySelector('div[data-testid="column"]:nth-of-type(2)');
        if (middleColumn) {
            // Find all buttons in the middle column
            const allButtons = middleColumn.querySelectorAll('button');
            allButtons.forEach(function(btn) {
                // Check if button is related to image (has fullscreen/expand attributes or is in image container)
                const isImageButton = btn.getAttribute('title') && btn.getAttribute('title').toLowerCase().includes('fullscreen') ||
                                     btn.getAttribute('aria-label') && btn.getAttribute('aria-label').toLowerCase().includes('fullscreen') ||
                                     btn.getAttribute('aria-label') && btn.getAttribute('aria-label').toLowerCase().includes('expand') ||
                                     btn.closest('[data-testid="stImage"]');
                
                if (isImageButton) {
                    // Aggressively hide and remove
                    btn.style.display = 'none';
                    btn.style.visibility = 'hidden';
                    btn.style.opacity = '0';
                    btn.style.pointerEvents = 'none';
                    btn.style.width = '0';
                    btn.style.height = '0';
                    btn.style.padding = '0';
                    btn.style.margin = '0';
                    btn.style.position = 'absolute';
                    btn.style.left = '-9999px';
                    // Remove from DOM
                    try {
                        btn.remove();
                    } catch(e) {}
                }
            });
            
            // Also specifically target image containers
            const logoImages = middleColumn.querySelectorAll('[data-testid="stImage"]');
            logoImages.forEach(function(imgContainer) {
                const buttons = imgContainer.querySelectorAll('button');
                buttons.forEach(function(btn) {
                    btn.style.display = 'none';
                    btn.style.visibility = 'hidden';
                    btn.style.opacity = '0';
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
    
    // Run immediately and more frequently
    removeLogoButtons();
    setTimeout(removeLogoButtons, 50);
    setTimeout(removeLogoButtons, 100);
    setTimeout(removeLogoButtons, 200);
    setTimeout(removeLogoButtons, 500);
    setInterval(removeLogoButtons, 200);
    
    // Watch for new elements with more aggressive settings
    const observer = new MutationObserver(function(mutations) {
        removeLogoButtons();
    });
    observer.observe(document.body, { 
        childList: true, 
        subtree: true,
        attributes: true,
        attributeFilter: ['style', 'class']
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Logo - Centered using columns
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.empty()
    with col2:
        try:
            st.image("assets/Logo.svg", width=280)
        except:
            st.image("assets/Logo.svg")
    with col3:
        st.empty()
    
    # Header
    st.markdown('<h1 class="main-header">Schema Documentation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Understanding data types and schemas for BigQuery</p>', unsafe_allow_html=True)
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back to Main Page", use_container_width=True, key="back_to_main"):
            st.switch_page("app.py")
    with col2:
        st.empty()
    with col3:
        if st.button("View BigQuery Guide →", use_container_width=True, key="nav_to_docs"):
            st.switch_page("pages/Documentation.py")
    
    # Data Types Quick Reference Table
    st.markdown("""
    <div class="info-card">
        <h3>📊 Data Types Quick Reference</h3>
        <p>Here's a quick overview of all available data types:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create the table using Streamlit
    data_types_table = {
        "Data Type": ["STRING", "INT64", "FLOAT64", "BOOL", "DATE", "TIMESTAMP"],
        "Use For": [
            "Text, names, addresses, IDs with letters",
            "Whole numbers (rarely used - prefer FLOAT64)",
            "All numbers (quantities, prices, counts, measurements)",
            "True/False or 0/1 values (Yes/No questions)",
            "Calendar dates without time",
            "Dates with specific time information"
        ],
        "Example": [
            '"Product-123", "John Doe", "user@email.com"',
            "42, 2024, 150, -10",
            "29.99, 42, 150, 1.5, 87.5",
            "TRUE/FALSE or 0/1",
            "2024-03-15",
            "2024-03-15 14:30:00"
        ]
    }
    
    df_table = pd.DataFrame(data_types_table)
    st.dataframe(df_table, use_container_width=True, hide_index=True)
    
    # Important note about numbers
    st.markdown("""
    <div class="important-notice" style="margin-top: 1.5rem;">
        <h3>🔢 Important: Number Data Types</h3>
        <p><strong>For any numbers, always select FLOAT64.</strong></p>
        <p>This includes quantities, prices, sales, measurements, counts, percentages, and any numeric values - use FLOAT64 for all of them.</p>
        <p><strong>Exception:</strong> If your column contains only 0 and 1 values, use BOOL instead.</p>
        <p>Why FLOAT64 for all numbers?</p>
        <ul>
            <li>Ensures accurate calculations and prevents rounding errors</li>
            <li>Supports decimal precision when needed</li>
            <li>Handles both whole numbers and decimals seamlessly</li>
            <li>Provides consistent data type handling across all numeric columns</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="info-card">
        <h3>📚 What is a Data Schema?</h3>
        <p>A <strong>data schema</strong> defines the structure of your data - it tells BigQuery what type of information each column contains. Think of it like a blueprint that describes your data before you upload it.</p>
        <p>When you upload data to BigQuery, you need to specify the data type for each column. This helps BigQuery:</p>
        <ul>
            <li>Store your data efficiently</li>
            <li>Perform accurate calculations and queries</li>
            <li>Validate data quality</li>
            <li>Optimize performance</li>
        </ul>
        <p><strong>Our app automatically detects the best data type for each column, but you can review and change them before processing!</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # STRING Type
    st.markdown("""
    <div class="info-card">
        <h3><span class="type-badge">STRING</span> Text and Mixed Content</h3>
        <p><strong>Use for:</strong> Any text, names, addresses, IDs with letters, descriptions, or mixed alphanumeric content.</p>
        <p><strong>Characteristics:</strong></p>
        <ul>
            <li>Can contain letters, numbers, spaces, and special characters</li>
            <li>No mathematical operations can be performed</li>
            <li>Perfect for names, addresses, product codes, and descriptions</li>
            <li>Our app automatically detects STRING if a column contains any letters (A-Z)</li>
        </ul>
        <div class="example-box">
            <div class="example-box-title">Example Data:</div>Product Name: "Laptop Pro 15"
Customer ID: "CUST-12345"
Email: "user@example.com"
Address: "123 Main St, New York"
Notes: "Special order - handle with care"
        </div>
        <p><strong>When to use:</strong> Use STRING for any column that contains text, even if it looks like numbers (like phone numbers, postal codes, or IDs that start with zero).</p>
    </div>
    """, unsafe_allow_html=True)
    
    # INT64 Type
    st.markdown("""
    <div class="info-card">
        <h3><span class="type-badge">INT64</span> Whole Numbers</h3>
        <p><strong>Note:</strong> While INT64 is available, we recommend using FLOAT64 for all numbers instead. FLOAT64 provides better flexibility and consistency.</p>
        <p><strong>Characteristics:</strong></p>
        <ul>
            <li>Only whole numbers (integers) - no decimal points</li>
            <li>Can be positive, negative, or zero</li>
            <li>Can perform mathematical operations (addition, subtraction, etc.)</li>
        </ul>
        <div class="example-box">
            <div class="example-box-title">Example Data:</div>Quantity: 42
Year: 2024
Age: 35
Order Count: 150
Temperature: -10
        </div>
        <p><strong>⚠️ Important:</strong> For consistency and accuracy, use FLOAT64 for all numbers instead of INT64. The only exception is if your column contains only 0 and 1 values - in that case, use BOOL.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # FLOAT64 Type
    st.markdown("""
    <div class="info-card">
        <h3><span class="type-badge">FLOAT64</span> All Numbers</h3>
        <p><strong>Use for:</strong> All numbers - quantities, prices, sales, measurements, counts, percentages, years, ages, and any numeric values.</p>
        <p><strong>🔢 Important:</strong> Always use FLOAT64 for any numeric data, whether it has decimals or not. This ensures accurate calculations and consistent data handling.</p>
        <p><strong>Exception:</strong> If your column contains only 0 and 1 values, use BOOL instead.</p>
        <p><strong>Characteristics:</strong></p>
        <ul>
            <li>Can contain decimal points (like 3.14, 99.99, 0.5) or whole numbers (like 42, 100, 2024)</li>
            <li>Perfect for all numeric data - prices, quantities, measurements, percentages, counts, etc.</li>
            <li>Supports high precision calculations</li>
            <li>Can be positive, negative, or zero</li>
            <li>Provides consistent data type handling across all numeric columns</li>
        </ul>
        <div class="example-box">
            <div class="example-box-title">Example Data:</div>Price: 29.99
Quantity: 42
Weight: 1.5
Percentage: 87.5
Year: 2024
Count: 150
        </div>
        <p><strong>When to use:</strong> Use FLOAT64 for all numbers. This is the recommended choice for any column containing numeric values.</p>
        <p><strong>💡 Tip:</strong> Even if your data looks like whole numbers (like 100 or 42), use FLOAT64 for consistency and to ensure accurate calculations. The only exception is columns with only 0 and 1 values - use BOOL for those.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # BOOL Type
    st.markdown("""
    <div class="info-card">
        <h3><span class="type-badge">BOOL</span> True or False Values</h3>
        <p><strong>Use for:</strong> Yes/No questions, on/off status, true/false flags, binary choices, and columns containing only 0 and 1 values.</p>
        <p><strong>🔢 Important:</strong> If your column contains only 0 and 1 values, use BOOL instead of FLOAT64. This is the exception to the "use FLOAT64 for all numbers" rule.</p>
        <p><strong>Characteristics:</strong></p>
        <ul>
            <li>Only two possible values: TRUE or FALSE (or 0 and 1)</li>
            <li>Very efficient storage (takes minimal space)</li>
            <li>Perfect for flags, status indicators, and binary choices</li>
            <li>Our app recognizes common formats like: Yes/No, True/False, 1/0, Y/N</li>
        </ul>
        <div class="example-box">
            <div class="example-box-title">Example Data:</div>Is Active: TRUE (or 1)
Has Account: FALSE (or 0)
Is Verified: TRUE (or 1)
Is Premium: FALSE (or 0)
Completed: TRUE (or 1)
        </div>
        <p><strong>When to use:</strong> Use BOOL when your column represents a simple yes/no, on/off, or true/false question, or when it contains only 0 and 1 values. This is much more efficient than using STRING with "Yes"/"No" values or FLOAT64 for binary data.</p>
        <p><strong>💡 Tip:</strong> If your data uses "Yes"/"No" or "1"/"0", our app will automatically convert them to TRUE/FALSE.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # DATE Type
    st.markdown("""
    <div class="info-card">
        <h3><span class="type-badge">DATE</span> Calendar Dates</h3>
        <p><strong>Use for:</strong> Birthdays, order dates, deadlines, and any date that doesn't include time information.</p>
        <p><strong>Characteristics:</strong></p>
        <ul>
            <li>Stores only the date (year, month, day) - no time information</li>
            <li>Format: YYYY-MM-DD (e.g., 2024-03-15)</li>
            <li>Perfect for dates when you don't need to know the exact time</li>
            <li>Can calculate differences between dates (like "days since order")</li>
        </ul>
        <div class="example-box">
            <div class="example-box-title">Example Data:</div>Birth Date: 1990-05-15
Order Date: 2024-03-20
Deadline: 2024-12-31
Start Date: 2024-01-01
        </div>
        <p><strong>When to use:</strong> Use DATE when you only need the calendar date without time information. For example: birthdays, order dates, deadlines, or any date-based filtering.</p>
        <p><strong>⚠️ Important:</strong> If your dates include time (like "2024-03-15 14:30:00"), use TIMESTAMP instead.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # TIMESTAMP Type
    st.markdown("""
    <div class="info-card">
        <h3><span class="type-badge">TIMESTAMP</span> Dates with Time</h3>
        <p><strong>Use for:</strong> Event timestamps, transaction times, log entries, and any date that includes specific time information.</p>
        <p><strong>Characteristics:</strong></p>
        <ul>
            <li>Stores both date and time (year, month, day, hour, minute, second)</li>
            <li>Format: YYYY-MM-DD HH:MM:SS (e.g., 2024-03-15 14:30:00)</li>
            <li>Perfect for tracking exact moments in time</li>
            <li>Can calculate time differences and perform time-based queries</li>
        </ul>
        <div class="example-box">
            <div class="example-box-title">Example Data:</div>Created At: 2024-03-15 14:30:00
Last Login: 2024-03-20 09:15:30
Transaction Time: 2024-03-25 16:45:12
Event Timestamp: 2024-04-01 12:00:00
        </div>
        <p><strong>When to use:</strong> Use TIMESTAMP when you need to know the exact time something happened. For example: when a user logged in, when an order was placed, or when a transaction occurred.</p>
        <p><strong>💡 Tip:</strong> If your dates don't include time information, use DATE instead - it's more efficient and easier to work with.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Best Practices
    st.markdown("""
    <div class="info-card">
        <h3>✅ Best Practices for Choosing Data Types</h3>
        <p><strong>1. Be Consistent:</strong> Use the same data type for similar data across your dataset. For example, all prices should be FLOAT64, all IDs should be STRING or INT64 consistently.</p>
        <p><strong>2. Choose the Right Precision:</strong></p>
        <ul>
            <li>Use INT64 for whole numbers (quantities, counts)</li>
            <li>Use FLOAT64 for decimal numbers (prices, measurements)</li>
            <li>Don't use FLOAT64 for whole numbers - INT64 is more efficient</li>
        </ul>
        <p><strong>3. Text vs Numbers:</strong></p>
        <ul>
            <li>If a column contains ANY letters, use STRING (even if it looks like a number)</li>
            <li>Phone numbers, postal codes, and IDs that start with zero should be STRING</li>
            <li>Only use numeric types (INT64/FLOAT64) if you plan to do calculations</li>
        </ul>
        <p><strong>4. Date vs Timestamp:</strong></p>
        <ul>
            <li>Use DATE if you only need the calendar date</li>
            <li>Use TIMESTAMP if you need the exact time something happened</li>
            <li>DATE is more efficient if you don't need time information</li>
        </ul>
        <p><strong>5. Review Before Processing:</strong> Always review the automatically detected types in the Schema Review section. Our app is smart, but you know your data best!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Common Mistakes
    st.markdown("""
    <div class="info-card">
        <h3>⚠️ Common Mistakes to Avoid</h3>
        <p><strong>1. Don't use text for numbers.</strong> If a value is meant for math like prices, counts or measurements, store it as a number type (FLOAT64) instead of text. Text cannot be used for calculations.</p>
        <p><strong>2. Use one single number type for all normal numeric values.</strong> Even if the value is a whole number, still use FLOAT64. This keeps everything consistent and avoids problems when numbers change later.</p>
        <p><strong>3. Only use true or false when the column is strictly yes or no.</strong> If a column only contains 0 and 1 and those values mean yes or no, set it as BOOL. If those numbers might ever be something else, use FLOAT64 instead.</p>
        <p><strong>4. Don't store dates as text.</strong> If the value is a date like 2024-05-12, use a proper date type (DATE or TIMESTAMP). This allows sorting and filtering by date without issues.</p>
        <p><strong>5. Use DATE when you only care about the day.</strong> If you do not need the time of day, choose DATE instead of TIMESTAMP. It's simpler, cleaner and faster.</p>
        <p><strong>6. Always check the schema before uploading.</strong> Automatic detection is not perfect. Take a quick look to confirm each column type is correct so you avoid errors later.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p style="font-size: 1rem; margin-bottom: 0.25rem;"><strong>BigQuery Data Processor</strong></p>
        <p style="font-size: 0.75rem; color: #9ca3af; margin-bottom: 0.5rem;">Powered by dashworx</p>
        <p style="font-size: 0.9rem; color: #888; margin-bottom: 0.75rem;">Transform your data into BigQuery-ready format | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

