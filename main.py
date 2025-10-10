import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

st.set_page_config(page_title="Netflix EDA Dashboard", layout="wide")
st.title("Netflix EDA Dashboard")
st.markdown("Explore the Netflix dataset with interactive visualizations and insights by **year, country, genre, rating, directors & actors, and duration analysis**.")
st.markdown("---")
st.markdown("### About the Dataset")
st.markdown("""
The Netflix dataset contains information about movies and TV shows available on Netflix as of 2021. It includes various attributes such as:
- Title
- Type (Movie or TV Show)
- Director
- Cast
- Country
- Date Added
- Release Year
- Rating
- Duration
- Genre (Listed In)
- Description : 
This dataset provides a comprehensive overview of Netflix's content library, allowing for in-depth analysis of trends and patterns in the streaming service's offerings.
""")
st.image("https://t4.ftcdn.net/jpg/03/48/81/77/240_F_348817789_25OWzJSmz8pbFOc8HRhxEeMpdYBPeu7X.jpg")

@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month
    df['day_added'] = df['date_added'].dt.day
    df['country'] = df['country'].fillna('Unknown')
    df['rating'] = df['rating'].fillna('Unknown')
    df['listed_in'] = df['listed_in'].str.split(', ')
    df.dropna(subset=['date_added'], inplace=True)
    return df

df = load_data()

st.subheader("Dataset Overview")
st.dataframe(df.head(10))

st.subheader("Statistical Summary")
st.write(df.describe(include='all').T)


st.sidebar.header("Filter Options")

type_filter = st.sidebar.multiselect(
    "Select Content Type",
    options=df['type'].unique(),
    default=df['type'].unique()
)

countries = st.sidebar.multiselect(
    "Select Country",
    options=sorted(df['country'].unique()),
    default=df['country'].unique()
)

year_min, year_max = int(df['release_year'].min()), int(df['release_year'].max())
year_range = st.sidebar.slider(
    "Select Release Year Range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max)
)

# Apply filters
filtered_df = df[
    (df['type'].isin(type_filter)) &
    (df['country'].isin(countries)) &
    (df['release_year'].between(year_range[0], year_range[1])) &
    (df['listed_in'].notnull()) &
    (df['rating'].notnull()) &
    (df['date_added'].notnull()) &
    (df['director'].notnull()) &
    (df['duration'].notnull()) &
    (df['description'].notnull()) &
    (df['cast'].notnull()) &
    (df['title'].notnull())
]

st.markdown(f"### Showing {len(filtered_df)} records after filtering")

# Key Metrics
st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Titles", len(filtered_df))
col2.metric("Total Movies", len(filtered_df[filtered_df['type'] == 'Movie']))
col3.metric("Total TV Shows", len(filtered_df[filtered_df['type'] == 'TV Show']))

st.image("https://wallpaperaccess.com/full/3447848.jpg")

# Visualizations
st.subheader("Visual Insights")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Yearly Trends", "Top Countries", "Genre Distribution", "Rating Distribution","Directors & Actors", "Duration Analysis"])

# Yearly Trends
with tab1:
    st.markdown("### Content Added by Release Year")
    plt.figure(figsize=(10,5))
    yearly_counts = filtered_df['release_year'].value_counts().sort_index()
    sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, color='crimson',linewidth=2.5)
    plt.xlabel("Release Year")
    plt.ylabel("Number of Titles")
    plt.xticks(rotation=45)
    plt.grid(True) 
    plt.title("Number of Titles Added by Release Year")
    st.pyplot(plt.gcf())
    plt.clf()
    
    st.markdown("### Content Added by Year Added to Netflix")
    plt.figure(figsize=(10,5))
    added_year_counts = filtered_df['release_year'].value_counts().sort_index()
    sns.barplot(x=added_year_counts.index, y=added_year_counts.values, color='blue',linewidth=3)
    plt.xlabel("Year Added to Netflix")
    plt.ylabel("Number of Titles")
    plt.xticks(rotation=60)
    plt.grid(True)
    plt.title("Number of Titles Added by Year Added to Netflix")
    st.pyplot(plt.gcf())
    plt.clf()
    
    st.write("Note: The 'Year Added to Netflix' chart may show fewer titles in recent years due to data cut-off dates. Most of the content is added in the years 2016-2020.")

# Top Countries
with tab2:
    st.markdown("### Top Countries by Number of Titles")
    plt.figure(figsize=(10,5))
    top_countries = filtered_df['country'].value_counts().head(10)
    sns.barplot(x=top_countries.values, y=top_countries.index, palette='viridis')
    plt.xlabel("Number of Titles")
    plt.ylabel("Country")
    plt.title("Top 10 Countries by Number of Titles")
    st.pyplot(plt.gcf())
    plt.clf()
    
    st.write("Note: Countries with fewer titles are grouped under 'Other' in the dataset. Most titles are from the United States, India, and the United Kingdom.")
    
    # Stacked barplot of Movies vs TV Shows by Country
    st.markdown("### Stacked Bar Plot of Movies vs TV Shows by Country")
    plt.figure(figsize=(12,6))
    country_type_stacked = filtered_df[filtered_df['country'].isin(top_countries.index)]
    country_type_stacked = country_type_stacked.groupby(['country', 'type']).size().unstack(fill_value=0)
    country_type_stacked.plot(kind='bar', stacked=True, colormap='Paired', figsize=(12,6))
    plt.xlabel("Country")
    plt.ylabel("Number of Titles")
    plt.title("Stacked Bar Plot of Movies vs TV Shows by Country")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()
    st.write("Note: The stacked bar plot provides a clear comparison of the number of Movies and TV Shows in each country. It highlights the dominance of Movies in countries like India and the United Kingdom, while the United States has a more balanced distribution between the two content types.")
    
# Genre Distribution
with tab3:
    st.markdown("### Genre Distribution")
    all_genres = filtered_df["listed_in"].explode().value_counts().head(10)
    plt.figure(figsize=(10,5))
    sns.barplot(x=all_genres.values, y=all_genres.index, palette='magma')
    plt.xlabel("Number of Titles")
    plt.ylabel("Genre")
    plt.title("Top 10 Genres")
    st.pyplot(plt.gcf())
    plt.clf()
    
    st.write("Note: Each title can belong to multiple genres, so the total count may exceed the number of unique titles. Popular genres include International Movies, Dramas, Comedies, Action & Adventure, and Documentaries.")
    
    # Countplot of top genres by type
    st.markdown("### Top Genres by Content Type")
    plt.figure(figsize=(12,6))
    top_genres = filtered_df["listed_in"].explode().value_counts().head(10).index
    genre_type_counts = filtered_df[filtered_df["listed_in"].apply(lambda x: any(genre in x for genre in top_genres))]
    genre_type_counts = genre_type_counts.explode("listed_in")
    sns.countplot(data=genre_type_counts[genre_type_counts["listed_in"].isin(top_genres)], x="listed_in", hue="type", palette='Set1')
    plt.xlabel("Genre")
    plt.ylabel("Number of Titles")
    plt.title("Top Genres by Content Type")
    plt.xticks(rotation=55)
    st.pyplot(plt.gcf())
    plt.clf()
    st.write("Note: This chart shows the distribution of top genres across Movies and TV Shows. Some genres like Dramas and Comedies are prevalent in both types, while others may be more specific to one type.")
    
    # Directors by Genre
    st.subheader("Top 10 Directors by Genre")
    selected_genre = st.selectbox("Select a Genre", options=sorted(filtered_df['listed_in'].explode().unique()), index=0)
    genre_directors = filtered_df[filtered_df['listed_in'].apply(lambda x: selected_genre in x)]['director'].dropna().value_counts().head(10)
    plt.figure(figsize=(10,5))
    sns.barplot(x=genre_directors.values, y=genre_directors.index, palette='cividis')
    plt.xlabel("Number of Titles")
    plt.ylabel("Director")
    plt.title(f"Top 10 Directors in {selected_genre} Genre")
    st.pyplot(plt.gcf())
    plt.clf()
    st.write(f"Note: Directors who frequently work within the {selected_genre} genre are highlighted here. This can indicate specialization or popularity within that genre. For instance, certain directors may be known for their work in Documentaries or Comedies.")
    
# Rating Distribution
with tab4:
    st.markdown("### Rating Distribution")
    plt.figure(figsize=(10,5))
    rating_counts = filtered_df['rating'].value_counts().head(10)
    sns.barplot(x=rating_counts.values, y=rating_counts.index, palette='viridis')
    plt.xlabel("Number of Titles")
    plt.ylabel("Rating")
    plt.title("Top 10 Ratings")
    st.pyplot(plt.gcf())
    plt.clf()
    
    st.write("Note: Ratings indicate the suitability of content for different age groups. Common ratings include TV-MA, TV-14, TV-PG, and R.")
    
    st.markdown("### Ratings by Content Type")
    plt.figure(figsize=(12,6))
    sns.countplot(data=filtered_df[filtered_df['rating'].isin(rating_counts.index   )], x='rating', hue='type', palette='Set2')
    plt.xlabel("Rating")
    plt.ylabel("Number of Titles")
    plt.title("Ratings by Content Type")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()
    st.write("Note: This chart shows how different ratings are distributed between Movies and TV Shows. For example, TV-MA rated content is more prevalent in TV Shows, while R-rated content is more common in Movies.")
    
# Directors & Actors
with tab5:
    st.markdown("### Top Directors")
    plt.figure(figsize=(10,5))
    top_directors = filtered_df['director'].dropna().value_counts().head(10)
    sns.barplot(x=top_directors.values, y=top_directors.index, palette='cubehelix')
    plt.xlabel("Number of Titles")
    plt.ylabel("Director")
    plt.title("Top 10 Directors")
    st.pyplot(plt.gcf())
    plt.clf()
    
    st.write("Note: Some directors have multiple titles on Netflix, showcasing their popularity and prolific work in the industry. Notable directors include Rajiv Chilaka, Marcus Raboy, and Youssef Chahine.")
    
    st.markdown("### Top Actors")
    plt.figure(figsize=(10,5))
    all_actors = filtered_df["cast"].dropna().str.split(', ').explode().value_counts().head(10)
    sns.barplot(x=all_actors.values, y=all_actors.index, palette='Spectral')
    plt.xlabel("Number of Titles")
    plt.ylabel("Actor")
    plt.title("Top 10 Actors")
    st.pyplot(plt.gcf())
    plt.clf()
    
    st.write("Note: Actors with multiple appearances on Netflix are highlighted here, reflecting their frequent collaborations with Netflix productions. Most featured actors include Anupam Kher, Shah Rukh Khan, and Om Puri.")
    
    # Scatter Plot of Directors vs Number of Titles
    st.markdown("### Directors vs Number of Titles")
    plt.figure(figsize=(10,5))
    sns.scatterplot(x=top_directors.values, y=top_directors.index, size=top_directors.values, legend=False, color='purple', alpha=0.6)
    plt.xlabel("Number of Titles")
    plt.ylabel("Director")
    plt.title("Directors vs Number of Titles")
    st.pyplot(plt.gcf())
    plt.clf()
    st.write("Note: The scatter plot emphasizes directors with a high number of titles on Netflix. The size of the points corresponds to the number of titles, making it easy to identify prolific directors. For instance, directors like Rajiv Chilaka and Marcus Raboy stand out with numerous titles.")
    

# Duration Analysis
with tab6:
    st.markdown("### Duration Distribution for Movies")
    plt.figure(figsize=(10,5))
    movie_durations = filtered_df[filtered_df['type'] == 'Movie']['duration'].str.replace(' min', '').astype(int)
    sns.histplot(movie_durations, bins=30, kde=True, color='green')
    plt.xlabel("Duration (minutes)")  
    plt.ylabel("Number of Movies")
    plt.title("Movie Duration Distribution")
    st.pyplot(plt.gcf())
    plt.clf()
    
    st.write("Note: Most movies have durations between 80 to 120 minutes, with a few outliers extending beyond 200 minutes, and Most common movie lengths are around 90 minutes.")
    
    st.markdown("### Duration Distribution for TV Shows")
    plt.figure(figsize=(10,5))
    tv_durations = filtered_df[filtered_df['type'] == 'TV Show']['duration'].str.replace('Season', '').str.replace(' Seasons', '').str.replace('s','').astype(int)
    sns.histplot(tv_durations, bins=30, kde=True, color='orange')
    plt.xlabel("Number of Seasons")
    plt.ylabel("Number of TV Shows")
    plt.title("TV Show Duration Distribution")
    st.pyplot(plt.gcf())
    plt.clf()  
    
    st.write("Note: Most TV shows have between 1 to 3 seasons, with a few long-running series having more than 10 seasons. The most common number of seasons is 1.")
    
    # Box Plot for Duration
    st.markdown("### Box Plot of Duration by Content Type")
    plt.figure(figsize=(10,5))
    filtered_df['duration_num'] = filtered_df['duration'].str.extract('(\d+)').astype(int)
    sns.boxplot(x='type', y='duration_num', data=filtered_df, palette='Set2')
    plt.xlabel("Content Type")
    plt.ylabel("Duration (minutes for Movies, seasons for TV Shows)")
    plt.title("Box Plot of Duration by Content Type")
    st.pyplot(plt.gcf())
    plt.clf()
    st.write("Note: The box plot illustrates the spread of durations for movies and TV shows. Movies tend to have a wider range of durations, while TV shows are more concentrated around lower season counts.")
    
    # Boxplot of duration by rating
    st.markdown("### Box Plot of Duration by Rating")
    plt.figure(figsize=(12,6))
    sns.boxplot(x='rating', y='duration_num', data=filtered_df, palette='Set3')
    plt.xlabel("Rating")
    plt.ylabel("Duration (minutes for Movies, seasons for TV Shows)")
    plt.title("Box Plot of Duration by Rating")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()
    st.write("Note: The box plot shows how duration varies across different content ratings. For instance, TV-MA rated content tends to have a wider range of durations compared to TV-G or TV-PG rated content.")
    
    # Scatter Plot of Duration vs Release Year
    st.markdown("### Scatter Plot of Duration vs Release Year")
    plt.figure(figsize=(10,5))
    sns.scatterplot(x='release_year', y='duration_num', hue='type', data=filtered_df, alpha=0.6)
    plt.xlabel("Release Year")
    plt.ylabel("Duration (minutes for Movies, seasons for TV Shows)")
    plt.title("Scatter Plot of Duration vs Release Year")
    plt.legend(title='Content Type')
    st.pyplot(plt.gcf())
    plt.clf()
    st.write("Note: The scatter plot indicates that there is no strong correlation between release year and duration for both movies and TV shows. However, it can be observed that newer content tends to have a wider range of durations.")
    
 # Correlation Heatmap
st.subheader("Correlation Heatmap")
plt.figure(figsize=(10,6))
numeric_cols = filtered_df.select_dtypes(include=np.number)
corr = numeric_cols.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features")
st.pyplot(plt.gcf())
plt.clf()

st.write("Note: The correlation heatmap shows relationships between numerical features. Strong correlations can indicate potential influences between variables. For example, 'release_year' and 'year_added' may show some correlation as newer titles are often added to Netflix.")

st.markdown("---")
# Data Table
st.subheader("Filtered Data Table")
with st.expander("About this Dashboard"):
    st.dataframe(filtered_df)
    
st.markdown("---")
st.image("https://t3.ftcdn.net/jpg/06/05/84/54/240_F_605845445_h12BQAUO7ftEOQ4vWergz18SAxquTpwd.jpg")
st.caption("Built with using Streamlit, Pandas, Matplotlib, Seaborn | Dataset: Netflix Titles")

st.requirements = """
streamlit
pandas
numpy
matplotlib
seaborn
"""
st.download_button("Download Requirements", st.requirements, file_name="requirements.txt")

st.write("made by Gourav Sarkar - GitHub : gouravsarkar22")