#---------------IMPORTING PACKAGES-------------------
import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib
import requests
import pickle
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn._loss import _loss  # Force-load missing module
from sklearn.cluster import KMeans
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from sklearn.preprocessing import StandardScaler
import shap
from lightgbm import LGBMRegressor
from shap import KernelExplainer

# Set page configuration
st.set_page_config(page_title='Airbnb Prediction App', page_icon='🏨')

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("listing_bert.csv")

airbnb_data = load_data()

#---------------SIDE BAR CONFIGURATION-------------------
st.sidebar.title("Singapore Airbnb Price Prediction")
st.sidebar.markdown(
    """
    This web app allows you to explore a **linear regression model** for predicting 
    Airbnb listing prices in **Singapore**. 
    
    - Use the **Explore** tab to analyze the distribution of current Airbnb listing prices based on various factors.
    - Use the **Predict** tab to input details about a property and estimate its listing price.

    
    Filters in the side bar only applies to the **Explore** page!!
    (●'◡'●)

    """
)

# Section: Location & Property Details
st.sidebar.subheader("🏠 Location & Property Details")
location = st.sidebar.selectbox("Select Region", ["All", "Central Region", "East Region", "West Region", "North Region", "North-East Region"], key="location")
room_type = st.sidebar.selectbox("Room Type", ["All", "Entire home/apt", "Private room", "Shared room"], key="room_type")

# Section: Booking Requirements
st.sidebar.subheader("📌 Booking Requirements")
min_nights = st.sidebar.slider("Minimum Nights Required", 1, 30, 2, 1)
reviews_per_month = st.sidebar.slider("Average Reviews per Month", 0.0, 15.0, 1.0, 0.1)
total_review_no = st.sidebar.slider("Total number of reviews", 0.0, 400.0, 1.0, 1.0)
availability365 = st.sidebar.slider("Days Available", 0.0, 365.0, 1.0, 1.0)

# Tabs
tab1, tab2 = st.tabs(['Explore', 'Predict'])


            
# Function to analyze categorical columns
def analyse_categorical(df, col):
    frequency_df = df[col].value_counts().reset_index()
    frequency_df.columns = [col, 'count']
    labels = frequency_df[col]
    counts = frequency_df['count']

    # Prepare data for the box plot (price values for each category)
    boxplot_data = []
    boxplot_categories = []
    grouped_price_data = df.groupby(col)['price'].apply(list)

    for category, price in grouped_price_data.items():
        boxplot_data.append(price)
        boxplot_categories.append(category)

    # Calculate average and median price for each category
    stats_price_df = df.groupby(col)['price'].agg(['mean', 'median']).reset_index()
    stats_price_df.columns = [col, 'avg_price', 'median_price']

    # Create plots using Streamlit
    st.markdown(
            f"""
            <div style="margin-top: 20px;"></div>
            <h5 style="text-align: center; white-space: nowrap;"> {col.replace('_', ' ').title()} Analysis </h5>
            """,
            unsafe_allow_html=True
            )
    
    # Frequency Plot
    st.markdown(f"##### 🔢 Listing frequency")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set(style="whitegrid", palette="pastel")
    bars = ax.barh(labels, counts, color=sns.color_palette())
    ax.set_xlabel("Frequency")
    ax.set_ylabel(col)
    ax.set_title(f"Frequency of {col}")

    # Add data labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{count}", va="center", fontsize=12)

    st.pyplot(fig)

    # Box Plot
    st.markdown(f"##### 💹 Price Distribution by {col.replace('_', ' ').title()}")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=boxplot_data, orient="h", ax=ax)
    ax.set_yticks(range(len(boxplot_categories)))
    ax.set_yticklabels(boxplot_categories)
    ax.set_xlabel("Price")
    ax.set_ylabel(col)
    ax.set_title(f"Box Plot: Price Distribution by {col}")
    st.pyplot(fig)

    # Average and Median Price Plot
    st.markdown(f"#####  💸Average & Median Price by {col.replace('_', ' ').title()}")
    fig, ax = plt.subplots(figsize=(10, 6))
    stats_price_df_melted = stats_price_df.melt(id_vars=[col], value_vars=['avg_price', 'median_price'], 
                                                var_name='Statistic', value_name='Price')
    barplot = sns.barplot(x="Price", y=col, hue="Statistic", data=stats_price_df_melted, ax=ax)
    ax.set_xlabel("Price")
    ax.set_ylabel(col)
    ax.set_title(f"Average and Median Price by {col}")

    # Add labels inside the bars
    for container in barplot.containers:
        barplot.bar_label(container, fmt="%.2f", label_type="edge", fontsize=12, padding=3)

    st.pyplot(fig)




# st.dataframe(filtered)
#---------------EXPLORE TAB-------------------
with tab1:
    # st.markdown(
    #         f"""
    #         <h3 style="white-space: nowrap;">{"📍 Map Visualization: Neighborhoods vs Price vs Room Type"}</h3>
    #         """,
    #         unsafe_allow_html=True
    #     )

        # Filter data based on user input
    filtered = airbnb_data.copy()
    if location != "All":
        filtered = filtered[filtered["neighbourhood_group"] == location]
    if room_type != "All":
        filtered = filtered[filtered["room_type"] == room_type]

        # Apply  filters for Minimum Nights and Reviews per Month
        filtered = filtered[filtered["minimum_nights"] >= min_nights]
        filtered = filtered[filtered["reviews_per_month"] >= reviews_per_month]
        filtered = filtered[filtered["number_of_reviews"] >= total_review_no]
        filtered = filtered[filtered["availability_365"] >= availability365]

    # Ensure filtered data is not empty
    if filtered.empty:
        st.warning("⚠️ No current listings found after applying filters. Try adjusting the filters.")
    else:
        custom_colors = {
        "Shared room": "mediumseagreen",
        "Entire home/apt": "indianred",
        "Private room": "royalblue"
        
        }   

        st.markdown(
                f"""
                <h3 style="text-align: center; white-space: nowrap;">📍 Airbnb Listings in Singapore</h3>
                """,
                unsafe_allow_html=True
                )
        
        # Create map visualization with custom colors
        fig_map = px.scatter_mapbox(
            filtered,
            lat="latitude",
            lon="longitude",
            hover_name="name",
            hover_data={"price": True, "host_name": True, "neighbourhood": True},
            color="room_type",
            size="price",
            size_max=30,
            zoom=10,
            title=" Listing Room Type vs Price ",
            color_discrete_map=custom_colors  # Apply custom colors
        )

        # Set layout of the map 
        fig_map.update_layout(
            mapbox_style="open-street-map",
            width=1000,  # Adjust width
            height=500,  # Adjust height
            margin=dict(l=10, r=10, t=50, b=50),  # Reduce margins

            # Customizing legend position and layout
            legend=dict(
                title="Room Type",
                orientation="h",  # Display legend horizontally
                x=0.5,  # Center horizontally
                y=-0.15,  # Move below the map
                xanchor="center",  # Center alignment
                yanchor="bottom",  # Anchor at the top of the legend
                bgcolor="rgba(255,255,255,0.7)",  # Semi-transparent background
            )
        )

        st.plotly_chart(fig_map)

        st.markdown("---")
        st.markdown(
                f"""
                <h4 style="text-align: center; white-space: nowrap;">📊 Airbnb Price Distributions</h4>
                """,
                unsafe_allow_html=True
                )

        # Create a single figure with two subplots (ax1 for histogram, ax2 for box plot)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # One row, two columns

        # Histogram Plot
        sns.histplot(filtered['price'], bins=50, color='lightblue', edgecolor='black', kde=True, ax=ax1)
        ax1.set_xlabel("Price (SGD)")
        ax1.set_ylabel("Number of Listings")
        ax1.set_title("Histogram")

        # Horizontal Box Plot
        sns.boxplot(x=filtered['price'], color='lightblue', ax=ax2)
        ax2.set_xlabel("Price (SGD)")
        ax2.set_title("Box Plot")

        # Adjust layout
        plt.tight_layout()

        # Display both plots together in Streamlit
        st.pyplot(fig)
    
        # Calculate statistics safely
        mean_price = filtered["price"].mean()
        median_price = filtered["price"].median()
        max_price = filtered["price"].max()
        min_price = filtered["price"].min()

        # Create a summary DataFrame
        summary_data = pd.DataFrame({
            "Mean Price (SGD)": [mean_price],
            "Median Price (SGD)": [median_price],
            "Highest Price (SGD)": [max_price],
            "Lowest Price (SGD)": [min_price]
        })

    
        st.write("📊 **Price Statistics**")
        st.dataframe(summary_data)

        st.markdown("---")
        highest_priced_listing = filtered[filtered["price"] == max_price]
        lowest_priced_listing = filtered[filtered["price"] == min_price]

        # Handle edge cases where no listings match the highest/lowest price
        if highest_priced_listing.empty:
            st.warning("⚠️ No highest priced listing found.")
        else:
            highest_priced_listing = highest_priced_listing.iloc[0]

        if lowest_priced_listing.empty:
            st.warning("⚠️ No lowest priced listing found.")
        else:
            lowest_priced_listing = lowest_priced_listing.iloc[0]

        # Display details of the highest and lowest priced listings only if available
        st.markdown(
            f"""
            <h4 style="text-align: center; white-space: nowrap;">🏠 Highest & Lowest Priced Listings Details</h4>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            if not highest_priced_listing.empty:
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid #ddd;
                        border-radius: 10px;
                        padding: 15px;
                        background-color: #f9f9f9;
                        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                    ">
                        <h4 style="text-align: center; color: #007bff;">Highest Priced Listing</h4>
                        <p><b>🏷 Name:</b> {highest_priced_listing['name']}</p>
                        <p><b>📍 Neighbourhood:</b> {highest_priced_listing['neighbourhood']}</p>
                        <p><b>💰 Price:</b> SGD {highest_priced_listing['price']}</p>
                        <p><b>🏠 Room Type:</b> {highest_priced_listing['room_type']}</p>
                        <p><b>👤 Host Name:</b> {highest_priced_listing['host_name']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with col2:
            if not lowest_priced_listing.empty:
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid #ddd;
                        border-radius: 10px;
                        padding: 15px;
                        background-color: #f9f9f9;
                        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                    ">
                        <h4 style="text-align: center; color: #dc3545;">Lowest Priced Listing</h4>
                        <p><b>🏷 Name:</b> {lowest_priced_listing['name']}</p>
                        <p><b>📍 Neighbourhood:</b> {lowest_priced_listing['neighbourhood']}</p>
                        <p><b>💰 Price:</b> SGD {lowest_priced_listing['price']}</p>
                        <p><b>🏠 Room Type:</b> {lowest_priced_listing['room_type']}</p>
                        <p><b>👤 Host Name:</b> {lowest_priced_listing['host_name']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("---")

        with st.expander("Categorical Factors Analysis"):
                  
            st.markdown(
                    f"""
                    
                    <h4 style="text-align: center; white-space: nowrap;">📊 Airbnb Data Categorical Analysis</h4>
                    """,
                    unsafe_allow_html=True
                    )
            # Select categorical column
            categorical_columns = ["room_type", "neighbourhood_group", ]  # Update based on dataset
            selected_col = st.selectbox("Select a categorical column:", categorical_columns)

            # Run analysis when user selects a column
            if selected_col:
                analyse_categorical(airbnb_data, selected_col)
    
#---------------AIR BNB PRICE PREDICTION ------------------
    with tab2:

         # Dictionary mapping locations to their respective models
        model_files = {
        "Central Region": "Central_listings_model.pkl",
        "East Region": "East_listings_model.pkl",
        "North Region": "north_listings_model.pkl",
        "North-East Region": "north_east_listings_model.pkl",
        "West Region": "west_listings_model.pkl"
        }
        st.markdown(
            f"""
            <h3 style="text-align: center; white-space: nowrap;"> 💰 Airbnb Price Prediction</h3>
            """,
            unsafe_allow_html=True
            )
     

        # User Inputs for Prediction

        st.markdown(
        """
        <h5 style="font-size:20px;">🏠 Location & Property types:</h5>
        """,
        unsafe_allow_html=True  
        )

     #neighbourhood group
        pred_location = st.pills("1.Select Location", list(model_files.keys()), key="pred_location")

     #room type
        pred_room_type = st.pills("2.Room Type", ["Entire home/apt", "Private room", "Shared room"], key="pred_room_type")
        
     #neighbourhood
        # st.dataframe(airbnb_data)       
    
        # Compute the mean price for each neighbourhood
        neighbourhood_mean_price = airbnb_data.groupby("neighbourhood")["price"].mean()

        # Map the mean price to each row based on the neighbourhood
        airbnb_data["neighbourhood_target_mean"] = airbnb_data["neighbourhood"].map(neighbourhood_mean_price)


        filtered_data = airbnb_data.copy()
        filtered_data = filtered_data[filtered_data["neighbourhood_group"] == pred_location]
        filtered_data = filtered_data[filtered_data["room_type"] == pred_room_type]
        neighbourhood_options = list(filtered_data["neighbourhood"].unique())
        pred_neighbourhood = st.selectbox( "Select Neighbourhood",  neighbourhood_options, key="pred_neighbourhood")
        
        st.markdown(
          """
        <div style="margin-top: 20px;"></div>
        <h5 style="font-size:20px;">🎫 Booking details:</h5>
        """,
        unsafe_allow_html=True  
        )

         
     #minimum nights
        pred_min_nights = st.number_input("Minimum Nights Required", 1, 30, 2, 1, key="pred_min_nights")

     # reviews per month
        pred_reviews_per_month = st.number_input("Average Reviews per Month", 0.0, 15.0, 1.0, 0.1, key="pred_reviews_per_month")
    
     # number of reviews
        pred_review_no = st.number_input("Total number of reviews", 0.0, 400.0, 1.0, 1.0, key="pred_review_no")

     # Availability 365
        pred_avail = st.number_input("Days Available ", 0.0,365.0, 1.0, 1.0, key="pred_avail")


     
        filtered_data = filtered_data[filtered_data["neighbourhood"] == pred_neighbourhood] 
        # filtered_data = filtered_data[filtered_data["minimum_nights"] >= pred_min_nights] 
        # filtered_data = filtered_data[filtered_data["reviews_per_month"] >= pred_reviews_per_month] 
        # filtered_data = filtered_data[filtered_data["number_of_reviews"] >= pred_review_no] 
      
    
        try:
# Get most common longitude and latitude
            if not filtered_data["longitude"].empty:
                pred_longitude = round(filtered_data["longitude"].mode()[0], 2)
            else:
                pred_longitude = 0  # Default value

            if not filtered_data["latitude"].empty:
                pred_latitude = round(filtered_data["latitude"].mode()[0], 2)
            else:
                pred_latitude = 0  # Default value

            # Get the most common host_id, if available
            if not filtered_data["host_id"].empty:
                pred_host_id = int(filtered_data["host_id"].mode()[0])
            else:
                pred_host_id = -1  # Placeholder ID
            
            if not filtered_data["calculated_host_listings_count"].empty and not filtered_data["calculated_host_listings_count"].mode().empty:
                pred_cal_host_lis = int(filtered_data["calculated_host_listings_count"].mode()[0])
            else:
                pred_cal_host_lis = 0  # Default value

            # Get location & name cluster mode (integer value)
            if not filtered_data["location_cluster"].empty:
                pred_loc_cluster = int(filtered_data["location_cluster"].mode()[0])
            else:
                pred_loc_cluster = -1  # Default cluster

            if not filtered_data["naming_cluster"].empty:
                pred_name_cluster = int(filtered_data["naming_cluster"].mode()[0])
            else:
                pred_name_cluster = -1  # Default cluster

            # Get encoded neighbourhood value
            if not filtered_data["neighbourhood_target_mean"].empty:
                pred_neighbourhood_enc = int(filtered_data["neighbourhood_target_mean"].mode()[0])
            else:
                pred_neighbourhood_enc = 0  # Default mean
            
            # Prepare input for model
            input_data = pd.DataFrame({
                "host_id": [pred_host_id],
                "latitude": [pred_latitude],
                "longitude": [pred_longitude],
                "minimum_nights": [pred_min_nights],
                "number_of_reviews": [pred_review_no],
                "reviews_per_month": [pred_reviews_per_month],
                "calculated_host_listings_count": [pred_cal_host_lis],
                "availability_365": [pred_avail],
                "location_cluster": [pred_loc_cluster],
                "naming_cluster": [pred_name_cluster],
                "room_type_Entire home/apt": [0],
                "room_type_Private room": [0],
                "room_type_Shared room": [0],
                "neighbourhood_encoded": [pred_neighbourhood_enc],
            })

            # Assign one-hot encoding for room type
            if pred_room_type == "Entire home/apt":
                input_data["room_type_Entire home/apt"] = 1
            elif pred_room_type == "Private room":
                input_data["room_type_Private room"] = 1
            elif pred_room_type == "Shared room":
                input_data["room_type_Shared room"] = 1


    # # -----------FOR CHECKING-------------
    #     st.dataframe(input_data)
            # st.dataframe(filtered_data)

            # X_west_train_scaled = pd.read_csv("X_west_train_scaled.csv")
            # st.dataframe(X_west_train_scaled)

            # st.write(input_data.columns.tolist()) 
            
    #     st.write(X_west_train_scaled.columns.tolist()) 
    #     st.write(len(X_west_train_scaled))
    # #--------------------------------


        # conduct scaling 
                
            scaler = StandardScaler()
        
            # Load the appropriate dataset based on user selection
            if pred_location == 'Central Region':
                x_training = pd.read_csv("X_central_train.csv")

            elif pred_location == 'West Region':
                x_training = pd.read_csv("X_west_train.csv")

            elif pred_location == 'North-East Region':
                x_training = pd.read_csv("X_north_east_train.csv")

            elif pred_location == 'North Region':
                x_training = pd.read_csv("X_north_train.csv")

            elif pred_location == 'East Region':
                x_training = pd.read_csv("X_east_train.csv")

            # # Apply Standard Scaling
            scaled_features = scaler.fit(x_training)

            input_data_scaled = scaler.transform(input_data)

            x_training_scaled = scaler.transform(x_training)

            
            # Convert back to DataFrame with original column names
            input_data_scaled = pd.DataFrame(input_data_scaled, columns=input_data.columns)
            x_training_scaled = pd.DataFrame(x_training_scaled, columns=x_training.columns)
            # st.dataframe(x_training_scaled)
            # st.dataframe(input_data_scaled)
            # st.write(x_training_scaled.columns.tolist()) 
            

                # Load the prediction model
            model_file = model_files.get(pred_location)

            with open(model_file, 'rb') as file:
                rf = pickle.load(file)

            st.markdown("----")
            # Button to Predict Price
            if st.button("🔮 Predict Price"):
        
                # Predict the price
                predicted_price = rf.predict(input_data_scaled)[0]

                # Display prediction
                st.success(f"💰 Estimated Airbnb Price: **${predicted_price:.2f} SGD** per night")

                 # Extract final estimator from StackingRegressor
                # if hasattr(rf, "final_estimator_"):
                #     model_to_explain = rf.final_estimator_
                # else:
                model_to_explain = rf  # Fallback to the full model if no final estimator exists

                # st.dataframe(x_training_scaled)
                # st.dataframe(input_data_scaled)
                
                # st.write(f"🔍 Model expects {model_to_explain.n_features_in_} features")
                # st.write(f"📊 Training Data (`x_training_scaled`) Shape: {x_training_scaled.shape}")
                # st.write(f"📥 Input Data (`input_data_scaled`) Shape: {input_data_scaled.shape}")
                # st.write("🔍 Features the model was trained on:", model_to_explain.feature_names_in_)

                # # Define a prediction function for SHAP
                # def model_predict(X):
                #     return model_to_explain.predict(X)

                # shap_values = ""

                if pred_location == 'West Region':
                    explainer = KernelExplainer(model_to_explain.predict, x_training_scaled)
                    shap_values = explainer(x_training_scaled)

                else:
                # # Apply SHAP only to the final estimator
                    explainer = shap.Explainer(model_to_explain, x_training_scaled)
                    shap_values = explainer(x_training_scaled)
                    
                # st.write(shap_values)
                
                
                # Extract the SHAP values for the input prediction
                shap_values_single = shap_values[0]  # First row corresponds to the input_data prediction

                # Get the top contributing features (both positive and negative impact)
                top_positive_features = sorted(
                    zip(x_training_scaled.columns, shap_values_single.values),
                    key=lambda x: x[1], reverse=True
                )[:3]  # Top 3 positive drivers

                top_negative_features = sorted(
                    zip(x_training_scaled.columns, shap_values_single.values),
                    key=lambda x: x[1]
                )[:3]  # Top 3 negative drivers

                # only show the explanation when the user wants to see it
                with st.expander("View Explanation"):
                  
                    st.markdown(
                        f"""
                        <h3 style="text-align: center; white-space: nowrap;"> 💰 Understanding the Prediction Model</h3>
                        """,
                        unsafe_allow_html=True
                        )
                  
                    # Generate a text explanation
                    explanation = f"""
                    📊 The **SHAP charts** below explain what influenced the predicted Airbnb price for listings in {pred_location}.

                    🔹 **Key Factors Increasing Price:**
                    - {top_positive_features[0][0].replace('_', ' ')}
                    - {top_positive_features[1][0].replace('_', ' ')}
                    - {top_positive_features[2][0].replace('_', ' ')}

                    🔻 **Key Factors Decreasing Price:**
                    - {top_negative_features[0][0].replace('_', ' ')}
                    - {top_negative_features[1][0].replace('_', ' ')}
                    - {top_negative_features[2][0].replace('_', ' ')}

                     """

                    # Display the explanation before the charts
                    st.markdown(explanation)

                    # SHAP Waterfall Plot
                    st.markdown(
                    """
                    <div style="margin-top: 20px;"></div>
                    <h5 style="font-size:20px;">📌 Price Calculation Breakdown: </h5>
                    """,
                    unsafe_allow_html=True  
                    )

                    fig_waterfall = plt.figure(figsize=(10, 6))
                    shap.plots.waterfall(shap_values[0])
                    st.pyplot(fig_waterfall)

                    # SHAP Summary Plot

                    st.markdown(
                    """
                    <div style="margin-top: 20px;"></div>
                    <h5 style="font-size:20px;">📊 What Influences Airbnb Prices?</h5>
                    """,
                    unsafe_allow_html=True  
                    )

                    fig_summary, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values, x_training_scaled, show=False)
                    st.pyplot(fig_summary)

                    st.markdown(
                    """
                    <div style="margin-top: 20px;"></div>
                    <h5 style="font-size:20px;">💡 Overview:</h5>
                    """,
                    unsafe_allow_html=True  
                    )

                    
                    st.markdown("""
                    - Factors like **room type, location, or host credibility** tends to **increase the price** the most.
                    - When **availability is low or there are fewer reviews**, the listing might be **less competitive**, leading to **lower prices**.
                   """)
                
                            
            # SHAP Waterfall Plot (Fix: Use Matplotlib figure)
        except NameError as e:
            if "'x_training' is not defined" in str(e):
                st.warning("⚠️ Please select a **region and room type** before making a prediction.")
            else:
                st.error(f"⚠️ An error occurred: {e}")
        
        except Exception as e:
            st.write(e)
