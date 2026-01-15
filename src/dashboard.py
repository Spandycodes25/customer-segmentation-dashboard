import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Page config
st.set_page_config(
    page_title="Customer Segmentation Analysis",
    page_icon="👥",
    layout="wide"
)

# Title
st.title("👥 E-Commerce Customer Segmentation Dashboard")
st.markdown("""
Analyzed **800k+ transactions** from 5,878 customers to identify distinct segments using RFM (Recency, Frequency, Monetary) analysis and K-means clustering.
""")

# Load data
@st.cache_data
def load_data():
    rfm = pd.read_csv('data/processed/rfm_segmented.csv')
    return rfm

rfm = load_data()

# Segment names
segment_names = {0: 'Lost Customers', 1: 'Core Customers', 2: 'VIP Champions'}
rfm['SegmentName'] = rfm['Cluster'].map(segment_names)

# Sidebar
st.sidebar.header("📊 Dashboard Controls")
st.sidebar.markdown("---")

# Segment filter
selected_segments = st.sidebar.multiselect(
    "Select Segments to Display",
    options=list(segment_names.values()),
    default=list(segment_names.values())
)

# Filter data
filtered_rfm = rfm[rfm['SegmentName'].isin(selected_segments)]

st.sidebar.markdown("---")
st.sidebar.metric("Total Customers", f"{len(rfm):,}")
st.sidebar.metric("Total Revenue", f"${rfm['Monetary'].sum():,.0f}")
st.sidebar.metric("Avg Customer Value", f"${rfm['Monetary'].mean():,.0f}")

# Main dashboard tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔍 Segment Analysis", "💡 Business Insights", "📊 Interactive 3D"])

# TAB 1: Overview
with tab1:
    st.header("Customer Segmentation Overview")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(filtered_rfm):,}")
    with col2:
        st.metric("Segments", len(filtered_rfm['SegmentName'].unique()))
    with col3:
        st.metric("Total Revenue", f"${filtered_rfm['Monetary'].sum()/1e6:.2f}M")
    with col4:
        avg_recency = filtered_rfm['Recency'].mean()
        st.metric("Avg Recency", f"{avg_recency:.0f} days")
    
    st.markdown("---")
    
    # Revenue and customer distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue by Segment")
        revenue_by_segment = filtered_rfm.groupby('SegmentName')['Monetary'].sum().reset_index()
        fig = px.bar(revenue_by_segment, x='SegmentName', y='Monetary',
                    color='SegmentName',
                    color_discrete_map={'Lost Customers': 'red', 'Core Customers': 'green', 'VIP Champions': 'gold'},
                    labels={'Monetary': 'Total Revenue ($)', 'SegmentName': 'Segment'})
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Customer Distribution")
        customer_counts = filtered_rfm['SegmentName'].value_counts()
        fig = px.pie(values=customer_counts.values, names=customer_counts.index,
                    color=customer_counts.index,
                    color_discrete_map={'Lost Customers': 'red', 'Core Customers': 'green', 'VIP Champions': 'gold'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # RFM distributions
    st.markdown("---")
    st.subheader("RFM Metric Distributions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.histogram(filtered_rfm, x='Recency', nbins=50, 
                          title='Recency Distribution',
                          labels={'Recency': 'Days Since Last Purchase'})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        freq_capped = filtered_rfm['Frequency'].clip(upper=50)
        fig = px.histogram(freq_capped, x='Frequency', nbins=30,
                          title='Frequency Distribution (capped at 50)',
                          labels={'Frequency': 'Number of Purchases'})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        monetary_capped = filtered_rfm['Monetary'].clip(upper=10000)
        fig = px.histogram(monetary_capped, x='Monetary', nbins=50,
                          title='Monetary Distribution (capped at $10k)',
                          labels={'Monetary': 'Total Spending ($)'})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# TAB 2: Segment Analysis
with tab2:
    st.header("Detailed Segment Analysis")
    
    # Segment selector
    selected_segment = st.selectbox("Select a segment to analyze", 
                                    options=list(segment_names.values()))
    
    segment_data = filtered_rfm[filtered_rfm['SegmentName'] == selected_segment]
    
    # Segment KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Customers", f"{len(segment_data):,}")
    with col2:
        st.metric("Avg Recency", f"{segment_data['Recency'].mean():.0f} days")
    with col3:
        st.metric("Avg Frequency", f"{segment_data['Frequency'].mean():.1f}")
    with col4:
        st.metric("Avg Monetary", f"${segment_data['Monetary'].mean():,.0f}")
    
    st.markdown("---")
    
    # Segment characteristics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("RFM Statistics")
        stats = segment_data[['Recency', 'Frequency', 'Monetary']].describe()
        st.dataframe(stats.style.format("{:.2f}"))
    
    with col2:
        st.subheader("Top 10 Customers in Segment")
        top_customers = segment_data.nlargest(10, 'Monetary')[['CustomerID', 'Recency', 'Frequency', 'Monetary']]
        st.dataframe(top_customers.style.format({'Monetary': '${:,.0f}'}))
    
    st.markdown("---")
    
    # Scatter plots
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(segment_data, x='Recency', y='Monetary',
                        color='Frequency', size='Monetary',
                        title='Recency vs Monetary',
                        labels={'Recency': 'Days Since Last Purchase', 'Monetary': 'Total Spending ($)'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(segment_data, x='Frequency', y='Monetary',
                        color='Recency', size='Monetary',
                        title='Frequency vs Monetary',
                        labels={'Frequency': 'Number of Purchases', 'Monetary': 'Total Spending ($)'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# TAB 3: Business Insights
with tab3:
    st.header("💡 Business Insights & Recommendations")
    
    # Calculate key metrics
    total_revenue = rfm['Monetary'].sum()
    
    st.markdown("### 📊 Key Findings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔴 Lost Customers")
        lost = rfm[rfm['SegmentName'] == 'Lost Customers']
        st.write(f"**{len(lost):,} customers** (38.8%)")
        st.write(f"**${lost['Monetary'].sum()/1e6:.1f}M revenue** (7%)")
        st.write(f"**Avg recency:** {lost['Recency'].mean():.0f} days")
        st.markdown("""
        **Strategy:**
        - Win-back email campaign with 20% discount
        - Survey to understand why they left
        - Low priority - already churned
        """)
    
    with col2:
        st.markdown("#### 🟢 Core Customers")
        core = rfm[rfm['SegmentName'] == 'Core Customers']
        st.write(f"**{len(core):,} customers** (60.8%)")
        st.write(f"**${core['Monetary'].sum()/1e6:.1f}M revenue** (75%)")
        st.write(f"**Avg frequency:** {core['Frequency'].mean():.1f} purchases")
        st.markdown("""
        **Strategy:**
        - Loyalty program to increase frequency
        - Personalized product recommendations
        - Early access to sales
        - **PROTECT THIS SEGMENT!**
        """)
    
    with col3:
        st.markdown("#### 💎 VIP Champions")
        vip = rfm[rfm['SegmentName'] == 'VIP Champions']
        st.write(f"**{len(vip):,} customers** (0.4%)")
        st.write(f"**${vip['Monetary'].sum()/1e6:.1f}M revenue** (18%)")
        st.write(f"**Avg spend:** ${vip['Monetary'].mean():,.0f}")
        st.markdown("""
        **Strategy:**
        - VIP account manager
        - Exclusive previews & private sales
        - Premium support
        - **Call if inactive >30 days!**
        """)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Critical Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **💰 Revenue Concentration**
        
        0.4% of customers (24 VIPs) generate 18% of total revenue ($3.2M).
        
        **Losing 1 VIP = Losing 149 regular customers!**
        """)
    
    with col2:
        st.warning("""
        **⚠️ Churn Risk**
        
        38.8% of customers haven't purchased in 400+ days.
        
        **Win-back campaign could recover $1.2M in dormant customers.**
        """)
    
    st.markdown("---")
    
    # Revenue comparison
    st.markdown("### 📈 Segment Comparison")
    
    comparison = rfm.groupby('SegmentName').agg({
        'CustomerID': 'count',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'sum']
    }).round(2)
    
    comparison.columns = ['Customer Count', 'Avg Recency (days)', 'Avg Frequency', 'Avg Monetary ($)', 'Total Revenue ($)']
    comparison['Revenue %'] = (comparison['Total Revenue ($)'] / comparison['Total Revenue ($)'].sum() * 100).round(1)
    
    st.dataframe(comparison.style.format({
        'Customer Count': '{:,.0f}',
        'Avg Recency (days)': '{:.0f}',
        'Avg Frequency': '{:.1f}',
        'Avg Monetary ($)': '${:,.0f}',
        'Total Revenue ($)': '${:,.0f}',
        'Revenue %': '{:.1f}%'
    }))

# TAB 4: Interactive 3D
with tab4:
    st.header("Interactive 3D Customer Segmentation")
    
    st.markdown("""
    **Instructions:** 
    - 🖱️ Click and drag to rotate
    - 🔍 Scroll to zoom
    - 👆 Hover over points to see customer details
    """)
    
    # Create 3D plot
    colors_map = {'Lost Customers': 'red', 'Core Customers': 'green', 'VIP Champions': 'gold'}
    
    fig = go.Figure()
    
    for segment_name in filtered_rfm['SegmentName'].unique():
        segment_data = filtered_rfm[filtered_rfm['SegmentName'] == segment_name]
        
        hover_text = [
            f"Customer ID: {cid}<br>" +
            f"Recency: {r} days<br>" +
            f"Frequency: {f} purchases<br>" +
            f"Monetary: ${m:,.0f}<br>" +
            f"Segment: {segment_name}"
            for cid, r, f, m in zip(
                segment_data['CustomerID'],
                segment_data['Recency'],
                segment_data['Frequency'],
                segment_data['Monetary']
            )
        ]
        
        fig.add_trace(go.Scatter3d(
            x=segment_data['Recency'],
            y=segment_data['Frequency'],
            z=np.log1p(segment_data['Monetary']),
            mode='markers',
            name=segment_name,
            marker=dict(
                size=4,
                color=colors_map[segment_name],
                opacity=0.7,
                line=dict(width=0.5, color='white')
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>'
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Recency (days)',
            yaxis_title='Frequency (purchases)',
            zaxis_title='Monetary (log scale)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        height=700,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Project:** E-Commerce Customer Segmentation | **Method:** RFM Analysis + K-means Clustering | **Dataset:** 800k+ transactions, 5,878 customers")