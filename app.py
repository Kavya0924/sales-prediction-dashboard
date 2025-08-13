import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Generate sample sales data
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    
    # Date range
    start_date = datetime.now() - timedelta(days=730)
    dates = pd.date_range(start=start_date, periods=730, freq='D')
    
    # Product categories
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    data = []
    for i, date in enumerate(dates):
        for _ in range(np.random.randint(5, 15)):  # 5-15 sales per day
            category = np.random.choice(categories)
            region = np.random.choice(regions)
            
            # Create seasonal patterns
            month = date.month
            day_of_week = date.weekday()
            
            # Base sales influenced by category, season, and day of week
            base_sales = {
                'Electronics': 500,
                'Clothing': 300,
                'Home & Garden': 200,
                'Sports': 250,
                'Books': 150
            }[category]
            
            # Seasonal multiplier
            seasonal_multiplier = 1.2 if month in [11, 12] else 1.0  # Holiday boost
            weekend_multiplier = 1.1 if day_of_week >= 5 else 1.0  # Weekend boost
            
            # Add some randomness
            sales_amount = base_sales * seasonal_multiplier * weekend_multiplier * np.random.uniform(0.5, 1.5)
            quantity = max(1, int(np.random.normal(3, 1)))
            
            data.append({
                'date': date,
                'category': category,
                'region': region,
                'sales_amount': round(sales_amount, 2),
                'quantity': quantity,
                'month': month,
                'day_of_week': day_of_week,
                'is_weekend': day_of_week >= 5
            })
    
    return pd.DataFrame(data)

# Load and prepare data
@st.cache_data
def prepare_data():
    df = generate_sample_data()
    
    # Feature engineering
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Encode categorical variables
    le_category = LabelEncoder()
    le_region = LabelEncoder()
    
    df['category_encoded'] = le_category.fit_transform(df['category'])
    df['region_encoded'] = le_region.fit_transform(df['region'])
    
    return df, le_category, le_region

# Train ML models
@st.cache_data
def train_models(df):
    # Prepare features and target
    features = ['category_encoded', 'region_encoded', 'quantity', 'month', 
                'day_of_week', 'is_weekend', 'quarter', 'day_of_year']
    X = df[features]
    y = df['sales_amount']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    lr_model = LinearRegression()
    
    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    rf_pred = rf_model.predict(X_test)
    lr_pred = lr_model.predict(X_test)
    
    # Calculate metrics
    rf_metrics = {
        'MAE': mean_absolute_error(y_test, rf_pred),
        'MSE': mean_squared_error(y_test, rf_pred),
        'R2': r2_score(y_test, rf_pred)
    }
    
    lr_metrics = {
        'MAE': mean_absolute_error(y_test, lr_pred),
        'MSE': mean_squared_error(y_test, lr_pred),
        'R2': r2_score(y_test, lr_pred)
    }
    
    return rf_model, lr_model, rf_metrics, lr_metrics, X_test, y_test, rf_pred, lr_pred

# Main app
def main():
    st.markdown('<h1 class="main-header">📊 Sales Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Machine Learning Analytics for Business Intelligence")
    
    # Load data
    df, le_category, le_region = prepare_data()
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Select ML Model",
        ["Random Forest", "Linear Regression", "Model Comparison"]
    )
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=[df['date'].min().date(), df['date'].max().date()],
        min_value=df['date'].min().date(),
        max_value=df['date'].max().date()
    )
    
    # Category filter
    selected_categories = st.sidebar.multiselect(
        "Select Categories",
        options=df['category'].unique(),
        default=df['category'].unique()
    )
    
    # Region filter
    selected_regions = st.sidebar.multiselect(
        "Select Regions",
        options=df['region'].unique(),
        default=df['region'].unique()
    )
    
    # Filter data
    if len(date_range) == 2:
        df_filtered = df[
            (df['date'].dt.date >= date_range[0]) & 
            (df['date'].dt.date <= date_range[1]) &
            (df['category'].isin(selected_categories)) &
            (df['region'].isin(selected_regions))
        ]
    else:
        df_filtered = df[
            (df['category'].isin(selected_categories)) &
            (df['region'].isin(selected_regions))
        ]
    
    # Train models
    rf_model, lr_model, rf_metrics, lr_metrics, X_test, y_test, rf_pred, lr_pred = train_models(df_filtered)
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sales", f"${df_filtered['sales_amount'].sum():,.2f}")
    
    with col2:
        st.metric("Total Transactions", f"{len(df_filtered):,}")
    
    with col3:
        st.metric("Average Order Value", f"${df_filtered['sales_amount'].mean():.2f}")
    
    with col4:
        st.metric("Best Performing Category", df_filtered.groupby('category')['sales_amount'].sum().idxmax())
    
    # Model Performance Section
    st.markdown("## 🤖 Model Performance")
    
    if model_choice == "Random Forest":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Random Forest Metrics")
            st.write(f"**Mean Absolute Error:** {rf_metrics['MAE']:.2f}")
            st.write(f"**Mean Squared Error:** {rf_metrics['MSE']:.2f}")
            st.write(f"**R² Score:** {rf_metrics['R2']:.3f}")
        
        with col2:
            # Feature importance
            feature_names = ['Category', 'Region', 'Quantity', 'Month', 'Day of Week', 'Is Weekend', 'Quarter', 'Day of Year']
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title="Feature Importance - Random Forest")
            st.plotly_chart(fig, use_container_width=True)
    
    elif model_choice == "Linear Regression":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Linear Regression Metrics")
            st.write(f"**Mean Absolute Error:** {lr_metrics['MAE']:.2f}")
            st.write(f"**Mean Squared Error:** {lr_metrics['MSE']:.2f}")
            st.write(f"**R² Score:** {lr_metrics['R2']:.3f}")
        
        with col2:
            # Actual vs Predicted scatter plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=lr_pred, mode='markers', name='Predictions'))
            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                   y=[y_test.min(), y_test.max()], 
                                   mode='lines', name='Perfect Prediction'))
            fig.update_layout(title="Actual vs Predicted Sales", 
                            xaxis_title="Actual Sales", 
                            yaxis_title="Predicted Sales")
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # Model Comparison
        comparison_df = pd.DataFrame({
            'Model': ['Random Forest', 'Linear Regression'],
            'MAE': [rf_metrics['MAE'], lr_metrics['MAE']],
            'MSE': [rf_metrics['MSE'], lr_metrics['MSE']],
            'R2_Score': [rf_metrics['R2'], lr_metrics['R2']]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Model Comparison")
            st.dataframe(comparison_df, use_container_width=True)
        
        with col2:
            fig = px.bar(comparison_df, x='Model', y='R2_Score', 
                        title="R² Score Comparison")
            st.plotly_chart(fig, use_container_width=True)
    
    # Sales Analytics
    st.markdown("## 📈 Sales Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales by category
        category_sales = df_filtered.groupby('category')['sales_amount'].sum().reset_index()
        fig = px.pie(category_sales, values='sales_amount', names='category', 
                    title="Sales Distribution by Category")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sales by region
        region_sales = df_filtered.groupby('region')['sales_amount'].sum().reset_index()
        fig = px.bar(region_sales, x='region', y='sales_amount', 
                    title="Sales by Region")
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis
    st.markdown("## 📅 Time Series Analysis")
    
    daily_sales = df_filtered.groupby('date')['sales_amount'].sum().reset_index()
    fig = px.line(daily_sales, x='date', y='sales_amount', 
                  title="Daily Sales Trend")
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction Tool
    st.markdown("## 🔮 Sales Prediction Tool")
    st.markdown("Make predictions for new scenarios:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pred_category = st.selectbox("Category", df['category'].unique())
        pred_region = st.selectbox("Region", df['region'].unique())
    
    with col2:
        pred_quantity = st.number_input("Quantity", min_value=1, max_value=20, value=3)
        pred_month = st.selectbox("Month", range(1, 13), index=datetime.now().month-1)
    
    with col3:
        pred_day_of_week = st.selectbox("Day of Week", 
                                       ["Monday", "Tuesday", "Wednesday", "Thursday", 
                                        "Friday", "Saturday", "Sunday"])
        day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
                      "Friday": 4, "Saturday": 5, "Sunday": 6}
        pred_dow_num = day_mapping[pred_day_of_week]
    
    with col4:
        pred_quarter = st.selectbox("Quarter", [1, 2, 3, 4])
        pred_day_of_year = st.number_input("Day of Year", min_value=1, max_value=365, value=100)
    
    if st.button("Make Prediction"):
        # Prepare prediction data
        pred_data = pd.DataFrame({
            'category_encoded': [le_category.transform([pred_category])[0]],
            'region_encoded': [le_region.transform([pred_region])[0]],
            'quantity': [pred_quantity],
            'month': [pred_month],
            'day_of_week': [pred_dow_num],
            'is_weekend': [pred_dow_num >= 5],
            'quarter': [pred_quarter],
            'day_of_year': [pred_day_of_year]
        })
        
        rf_prediction = rf_model.predict(pred_data)[0]
        lr_prediction = lr_model.predict(pred_data)[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"Random Forest Prediction: ${rf_prediction:.2f}")
        with col2:
            st.success(f"Linear Regression Prediction: ${lr_prediction:.2f}")
    
    # Data Explorer
    st.markdown("## 🔍 Data Explorer")
    if st.checkbox("Show Raw Data"):
        st.dataframe(df_filtered.head(1000), use_container_width=True)
    
    # Download section
    st.markdown("## 📥 Download Data")
    csv = df_filtered.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name=f"sales_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
    initial_sidebar_state="expanded"
)

