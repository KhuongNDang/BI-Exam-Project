import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ============================================
# SETUP
# ============================================
plt.style.use('dark_background')
sns.set_style('darkgrid')

# Load data
df = pd.read_csv('data/danish_movies_clean.csv')

# Parse columns
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['decade'] = (df['release_date'].dt.year // 10 * 10)
for col in ['genre_ids', 'production_company_ids']:
    df[col] = df[col].apply(eval)

genre_mapping = {
    28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
    80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
    14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
    9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction',
    10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'
}

df['genre_names'] = df['genre_ids'].apply(
    lambda x: [genre_mapping.get(int(g), 'Unknown') for g in x]
)

# ============================================
# SIDEBAR NAVIGATION
# ============================================
st.title('Danish Cinema Analysis')
page = st.sidebar.selectbox(
    'Navigate',
    ['Data Explorer', 'Hypothesis Results', 'Model Performance', 'Model Prediction']
)

# ============================================
# PAGE 1 - DATA EXPLORER
# ============================================
if page == 'Data Explorer':
    st.header('Data Explorer')

    # Sidebar filters
    st.sidebar.header('Filters')
    search = st.sidebar.text_input('Search for a film by title')
    decades = sorted(df['decade'].dropna().unique().tolist())
    selected_decade = st.sidebar.selectbox(
        'Select Decade', ['All'] + [str(int(d)) for d in decades]
    )
    all_genres = sorted(set(
        g for genres in df['genre_names'] for g in genres if g != 'Unknown'
    ))
    selected_genre = st.sidebar.selectbox('Select Genre', ['All'] + all_genres)
    min_votes = st.sidebar.slider('Minimum Vote Count', 0, 100, 5)
    runtime_range = st.sidebar.slider('Runtime (minutes)', 0, 300, (0, 300))

    # Apply filters
    filtered_df = df.copy()
    if search:
        filtered_df = filtered_df[
            filtered_df['title'].str.contains(search, case=False, na=False)
        ]
    if selected_decade != 'All':
        filtered_df = filtered_df[filtered_df['decade'] == int(selected_decade)]
    if selected_genre != 'All':
        filtered_df = filtered_df[
            filtered_df['genre_names'].apply(lambda x: selected_genre in x)
        ]
    filtered_df = filtered_df[filtered_df['vote_count'] >= min_votes]
    filtered_df = filtered_df[
        (filtered_df['runtime'].isna()) |
        ((filtered_df['runtime'] >= runtime_range[0]) &
         (filtered_df['runtime'] <= runtime_range[1]))
    ]

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric('Total Films', len(filtered_df))
    col2.metric('Average Rating', f"{filtered_df['vote_average'].mean():.2f}")
    col3.metric('Median Rating', f"{filtered_df['vote_average'].median():.2f}")

    # Rating distribution
    st.subheader('Rating Distribution')
    fig, ax = plt.subplots()
    sns.histplot(filtered_df['vote_average'], bins=20, ax=ax)
    ax.set_xlabel('Vote Average')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    # Average rating by genre
    st.subheader('Average Rating by Genre')
    df_exploded = filtered_df.explode('genre_names')
    genre_ratings = df_exploded.groupby('genre_names')['vote_average'].mean().sort_values()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    genre_ratings.plot(kind='barh', ax=ax2)
    ax2.set_xlabel('Vote Average')
    st.pyplot(fig2)

    # Top 10 films
    st.subheader('Top 10 Highest Rated Films')
    top10 = filtered_df[filtered_df['vote_count'] >= 5].nlargest(10, 'vote_average')[
        ['title', 'release_date', 'genre_names', 'runtime', 'vote_average', 'vote_count']
    ]
    st.dataframe(top10)

    # Raw data
    st.subheader('Raw Data')
    st.dataframe(
        filtered_df[
            ['title', 'release_date', 'genre_names', 'runtime', 'vote_average', 'vote_count']
        ].head(20)
    )

# ============================================
# PAGE 2 - HYPOTHESIS RESULTS
# ============================================
elif page == 'Hypothesis Results':
    st.header('Hypothesis Results')

    # Summary table
    st.subheader('Summary')
    summary = pd.DataFrame({
        'Hypothesis': ['H1', 'H2', 'H3', 'H4'],
        'Description': ['Era of Production', 'Genre', 'Runtime', 'Production Company'],
        'Result': ['Rejected', 'Rejected', 'Partially Supported', 'Rejected'],
        'P-value': ['0.666', '0.000', '0.000', '0.635']
    })
    st.dataframe(summary)

    st.divider()

    # Hypothesis selector
    selected_h = st.selectbox('Select Hypothesis', ['H1', 'H2', 'H3', 'H4'])

    # H1
    if selected_h == 'H1':
        st.subheader('H1 — Era of Production')
        st.write('Films released after 1950 will have higher average ratings than earlier films.')

        cutoff = st.slider('Select cutoff year', 1900, 2000, 1950)
        df_rated = df[df['vote_count'] >= 5].copy()
        before = df_rated[df_rated['release_date'].dt.year < cutoff]['vote_average'].mean()
        after = df_rated[df_rated['release_date'].dt.year >= cutoff]['vote_average'].mean()

        col1, col2 = st.columns(2)
        col1.metric(f'Before {cutoff} Mean Rating', f'{before:.2f}')
        col2.metric(f'After {cutoff} Mean Rating', f'{after:.2f}')

        fig, ax = plt.subplots()
        df_rated.groupby(
            df_rated['release_date'].dt.year // 10 * 10
        )['vote_average'].mean().plot(kind='bar', ax=ax)
        ax.set_xlabel('Decade')
        ax.set_ylabel('Average Rating')
        ax.set_title('Average Rating by Decade')
        st.pyplot(fig)
        st.error('H1 Rejected — No significant difference found (p = 0.666)')

    # H2
    elif selected_h == 'H2':
        st.subheader('H2 — Genre')
        st.write('Drama and Documentary will receive higher ratings than Action and Comedy.')

        all_genres = sorted(set(
            g for genres in df['genre_names'] for g in genres if g != 'Unknown'
        ))
        col1, col2 = st.columns(2)
        genre1 = col1.selectbox('Genre 1', all_genres, index=all_genres.index('Drama'))
        genre2 = col2.selectbox('Genre 2', all_genres, index=all_genres.index('Action'))

        df_genre = df.explode('genre_names')
        g1_mean = df_genre[df_genre['genre_names'] == genre1]['vote_average'].mean()
        g2_mean = df_genre[df_genre['genre_names'] == genre2]['vote_average'].mean()

        col1.metric(f'{genre1} Average Rating', f'{g1_mean:.2f}')
        col2.metric(f'{genre2} Average Rating', f'{g2_mean:.2f}')

        fig, ax = plt.subplots(figsize=(10, 6))
        genre_means = df_genre.groupby('genre_names')['vote_average'].mean().sort_values()
        colors = ['red' if g in [genre1, genre2] else 'teal' for g in genre_means.index]
        genre_means.plot(kind='barh', ax=ax, color=colors)
        ax.set_xlabel('Vote Average')
        ax.set_title('Average Rating by Genre')
        st.pyplot(fig)
        st.error('H2 Rejected — Action and Family rated highest, Documentary rated lowest (F=26.662, p≈0.000)')

    # H3
    elif selected_h == 'H3':
        st.subheader('H3 — Runtime')
        st.write('Films with runtime between 90-120 minutes will receive higher ratings.')

        col1, col2 = st.columns(2)
        col1.metric('Pearson Correlation', '0.466')
        col2.metric('Spearman Correlation', '0.372')

        runtime_val = st.slider('Select Runtime (minutes)', 0, 300, 90)
        df_runtime = df.dropna(subset=['runtime'])
        nearby_films = df_runtime[
            (df_runtime['runtime'] >= runtime_val - 10) &
            (df_runtime['runtime'] <= runtime_val + 10)
        ]
        avg_rating = nearby_films['vote_average'].mean()
        count = len(nearby_films)

        col1, col2 = st.columns(2)
        col1.metric(
            f'Average Rating (±10 min of {runtime_val})',
            f'{avg_rating:.2f}' if count > 0 else 'N/A'
        )
        col2.metric('Films in range', count)

        bins = [0, 30, 60, 90, 120, 180, 300]
        labels = ['<30', '30-60', '60-90', '90-120', '120-180', '180+']
        df_runtime['runtime_group'] = pd.cut(df_runtime['runtime'], bins=bins, labels=labels)
        fig, ax = plt.subplots()
        df_runtime.groupby('runtime_group')['vote_average'].mean().plot(kind='bar', ax=ax)
        ax.set_xlabel('Runtime Group')
        ax.set_ylabel('Average Rating')
        ax.set_title('Average Rating by Runtime Group')
        st.pyplot(fig)
        st.warning('H3 Partially Supported — Longer films tend to rate higher (p≈0.000)')

    # H4
    elif selected_h == 'H4':
        st.subheader('H4 — Production Company')
        st.write('A small number of production companies will dominate highly rated films.')

        col1, col2 = st.columns(2)
        col1.metric('T-statistic', '-0.475')
        col2.metric('P-value', '0.635')

        top_n = st.slider('Select number of top companies to display', 5, 20, 10)
        top_companies = df.explode('production_company_ids').groupby(
            'production_company_ids'
        )['vote_average'].mean().sort_values(ascending=False).head(top_n)

        fig, ax = plt.subplots()
        top_companies.plot(kind='bar', ax=ax)
        ax.set_xlabel('Production Company ID')
        ax.set_ylabel('Average Rating')
        ax.set_title(f'Top {top_n} Production Companies by Average Rating')
        st.pyplot(fig)

        film_count = df.explode('production_company_ids').groupby(
            'production_company_ids'
        ).size().sort_values(ascending=False).head(top_n)
        fig2, ax2 = plt.subplots()
        film_count.plot(kind='bar', ax=ax2)
        ax2.set_xlabel('Production Company ID')
        ax2.set_ylabel('Number of Films')
        ax2.set_title(f'Top {top_n} Production Companies by Film Count')
        st.pyplot(fig2)
        st.error('H4 Rejected — No significant difference found (p = 0.635)')

# ============================================
# PAGE 3 - MODEL PERFORMANCE
# ============================================
elif page == 'Model Performance':
    st.header('Model Performance')

    # Interactive model selector
    selected_model = st.selectbox(
        'Select Model to Inspect',
        ['Decision Tree', 'Random Forest', 'RF Balanced', 'RF Default']
    )

    confusion_matrices = {
        'Decision Tree': np.array([[3, 0, 30], [0, 0, 6], [10, 2, 245]]),
        'Random Forest': np.array([[2, 0, 31], [0, 0, 6], [12, 0, 245]]),
        'RF Balanced': np.array([[1, 0, 32], [0, 0, 6], [11, 0, 246]]),
        'RF Default': np.array([[2, 0, 31], [0, 0, 6], [12, 0, 245]])
    }

    metrics = {
        'Decision Tree': {'Accuracy': 0.84, 'Weighted F1': 0.81, 'High F1': 0.13, 'Low F1': 0.00},
        'Random Forest': {'Accuracy': 0.83, 'Weighted F1': 0.80, 'High F1': 0.09, 'Low F1': 0.00},
        'RF Balanced': {'Accuracy': 0.82, 'Weighted F1': 0.79, 'High F1': 0.04, 'Low F1': 0.00},
        'RF Default': {'Accuracy': 0.84, 'Weighted F1': 0.80, 'High F1': 0.05, 'Low F1': 0.00}
    }

    reports = {
        'Decision Tree': {
            'High': {'Precision': 0.23, 'Recall': 0.09, 'F1': 0.13},
            'Low': {'Precision': 0.00, 'Recall': 0.00, 'F1': 0.00},
            'Medium': {'Precision': 0.87, 'Recall': 0.95, 'F1': 0.91}
        },
        'Random Forest': {
            'High': {'Precision': 0.14, 'Recall': 0.06, 'F1': 0.09},
            'Low': {'Precision': 0.00, 'Recall': 0.00, 'F1': 0.00},
            'Medium': {'Precision': 0.87, 'Recall': 0.95, 'F1': 0.91}
        },
        'RF Balanced': {
            'High': {'Precision': 0.06, 'Recall': 0.03, 'F1': 0.04},
            'Low': {'Precision': 0.00, 'Recall': 0.00, 'F1': 0.00},
            'Medium': {'Precision': 0.86, 'Recall': 0.94, 'F1': 0.90}
        },
        'RF Default': {
            'High': {'Precision': 0.10, 'Recall': 0.03, 'F1': 0.05},
            'Low': {'Precision': 0.00, 'Recall': 0.00, 'F1': 0.00},
            'Medium': {'Precision': 0.87, 'Recall': 0.96, 'F1': 0.91}
        }
    }

    # Metrics row
    st.subheader(f'{selected_model} — Performance Metrics')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Accuracy', f"{metrics[selected_model]['Accuracy']:.0%}")
    col2.metric('Weighted F1', f"{metrics[selected_model]['Weighted F1']:.2f}")
    col3.metric('High F1', f"{metrics[selected_model]['High F1']:.2f}")
    col4.metric('Low F1', f"{metrics[selected_model]['Low F1']:.2f}")

    # Detailed classification report toggle
    if st.checkbox('Show detailed classification report'):
        st.dataframe(pd.DataFrame(reports[selected_model]).T)

    st.divider()

    # Chart type toggle outside columns
    chart_type = st.radio(
        'Feature Importance Chart Type',
        ['Horizontal Bar', 'Bar'],
        horizontal=True
    )

    # Side by side confusion matrix and feature importance
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Confusion Matrix')
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(
            confusion_matrices[selected_model],
            annot=True, fmt='d',
            cmap='RdYlGn',
            xticklabels=['High', 'Low', 'Medium'],
            yticklabels=['High', 'Low', 'Medium'],
            ax=ax
        )
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        st.pyplot(fig)

    with col2:
        st.subheader('Feature Importance')
        importance = pd.Series({
            'Runtime': 0.39,
            'Decade': 0.29,
            'Production Company': 0.21,
            'Genre': 0.11
        })
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        colors = [
            'teal' if v == importance.max() else 'steelblue'
            for v in importance.sort_values()
        ]
        if chart_type == 'Horizontal Bar':
            importance.sort_values().plot(kind='barh', ax=ax2, color=colors)
        else:
            importance.sort_values().plot(kind='bar', ax=ax2, color=colors)
        ax2.set_title('Feature Importance')
        st.pyplot(fig2)

    st.divider()

    # Model comparison charts
    st.subheader('Model Comparison')
    col1, col2 = st.columns(2)
    models = ['Decision Tree', 'Random Forest', 'RF Balanced', 'RF Default']
    colors = ['teal' if m == selected_model else 'steelblue' for m in models]

    with col1:
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        accuracies = [0.84, 0.83, 0.82, 0.84]
        ax3.bar(models, accuracies, color=colors)
        ax3.set_ylim(0.75, 0.90)
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Accuracy Comparison')
        plt.xticks(rotation=15)
        st.pyplot(fig3)

    with col2:
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        f1_scores = [0.81, 0.80, 0.79, 0.80]
        ax4.bar(models, f1_scores, color=colors)
        ax4.set_ylim(0.75, 0.85)
        ax4.set_ylabel('Weighted F1')
        ax4.set_title('F1 Score Comparison')
        plt.xticks(rotation=15)
        st.pyplot(fig4)

    st.divider()

    # Clustering performance
    st.subheader('Clustering Performance')
    col1, col2 = st.columns(2)
    col1.metric('KMeans Silhouette Score', '0.254')
    col2.metric('Number of Clusters', '3')
    st.write('''
        A silhouette score of 0.254 indicates weak cluster separation.
        The selected features do not clearly divide films into distinct
        rating groups, which is consistent with the supervised learning results.
    ''')

    st.divider()

    # Model limitations
    st.subheader('Model Limitations')
    st.warning('''
        Both models suffer from class imbalance — the vast majority of films
        are rated as Medium, causing models to default to predicting Medium.
        Low rated films were never correctly predicted and High rated films
        were rarely identified. Future improvements could include SMOTE
        oversampling, additional features such as director and cast,
        or a regression approach instead of classification.
    ''')

# ============================================
# PAGE 4 - MODEL PREDICTION
# ============================================
elif page == 'Model Prediction':
    st.header('🎬 Film Rating Predictor')
    st.write('Select film characteristics below to predict whether a Danish film will be rated Low, Medium, or High.')

    # Train model
    le = LabelEncoder()
    df_ml = df[df['vote_count'] >= 5].copy()
    df_ml['genre'] = df_ml['genre_names'].apply(
        lambda x: x[0] if len(x) > 0 else 'Unknown'
    )
    df_ml['genre_encoded'] = le.fit_transform(df_ml['genre'])
    df_ml['decade_encoded'] = le.fit_transform(df_ml['decade'].astype(str))
    df_ml['production_company'] = df_ml['production_company_ids'].apply(
        lambda x: x[0] if len(x) > 0 else 0
    )
    df_ml['runtime'] = df_ml['runtime'].fillna(df_ml['runtime'].median())

    def rating_category(rating):
        if rating < 4:
            return 'Low'
        elif rating < 7:
            return 'Medium'
        else:
            return 'High'

    df_ml['rating_category'] = df_ml['vote_average'].apply(rating_category)
    X = df_ml[['genre_encoded', 'decade_encoded', 'production_company', 'runtime']]
    y = df_ml['rating_category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    st.divider()

    # User inputs
    st.subheader('🎯 Enter Film Characteristics')
    col1, col2, col3 = st.columns(3)

    all_genres = sorted(set(
        g for genres in df['genre_names'] for g in genres if g != 'Unknown'
    ))

    with col1:
        selected_genre = st.selectbox('🎭 Genre', all_genres)

    with col2:
        selected_decade = st.selectbox(
            '📅 Decade', sorted(df['decade'].dropna().unique().tolist())
        )

    with col3:
        selected_runtime = st.number_input(
            '⏱️ Runtime (minutes)', min_value=1, max_value=600, value=90
        )

    # Summary box
    st.info(f'''
        **Selected characteristics:**
        🎭 Genre: **{selected_genre}** &nbsp;&nbsp;
        📅 Decade: **{int(selected_decade)}s** &nbsp;&nbsp;
        ⏱️ Runtime: **{selected_runtime} minutes**
    ''')

    st.divider()

    # Similar films
    st.subheader('🎥 Similar Films in Dataset')
    similar = df[
        (df['genre_names'].apply(lambda x: selected_genre in x)) &
        (df['decade'] == selected_decade) &
        (df['vote_count'] >= 5)
    ].nlargest(5, 'vote_average')[
        ['title', 'release_date', 'runtime', 'vote_average', 'vote_count']
    ]
    if len(similar) > 0:
        st.dataframe(similar)
        avg_similar = similar['vote_average'].mean()
        col1, col2 = st.columns(2)
        col1.metric('Similar Films Found', len(similar))
        col2.metric('Average Rating of Similar Films', f'{avg_similar:.2f}')
    else:
        st.warning('No similar films found in dataset for this combination.')

    st.divider()

    # Predict button centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button('🔮 Predict Rating Category', use_container_width=True)

    if predict_button:
        genre_encoded = le.fit_transform([selected_genre])[0]
        decade_encoded = le.fit_transform([str(int(selected_decade))])[0]
        prediction = model.predict(
            [[genre_encoded, decade_encoded, 0, selected_runtime]]
        )
        proba = model.predict_proba(
            [[genre_encoded, decade_encoded, 0, selected_runtime]]
        )[0]
        classes = model.classes_

        st.divider()
        st.subheader('🎯 Prediction Result')

        col1, col2, col3 = st.columns(3)
        col1.metric('Genre', selected_genre)
        col2.metric('Decade', f'{int(selected_decade)}s')
        col3.metric('Runtime', f'{selected_runtime} min')

        st.write('')

        if prediction[0] == 'High':
            st.success('### 🌟 Predicted Rating Category: HIGH')
            st.write('This film is predicted to receive a rating above 7.0')
        elif prediction[0] == 'Medium':
            st.warning('### ⭐ Predicted Rating Category: MEDIUM')
            st.write('This film is predicted to receive a rating between 4.0 and 7.0')
        else:
            st.error('### ❌ Predicted Rating Category: LOW')
            st.write('This film is predicted to receive a rating below 4.0')

        st.divider()

        # Probability breakdown
        st.subheader('📊 Confidence Breakdown')
        proba_df = pd.DataFrame({
            'Category': classes,
            'Probability': proba
        }).sort_values('Probability', ascending=False)

        # Metric boxes in a row
        col1, col2, col3 = st.columns(3)
        for col, (_, row) in zip([col1, col2, col3], proba_df.iterrows()):
            col.metric(row['Category'], f"{row['Probability']:.1%}")

        # Chart below
        fig, ax = plt.subplots(figsize=(8, 3))
        colors = [
            'green' if c == 'High' else
            'orange' if c == 'Medium' else
            'red' for c in proba_df['Category']
        ]
        ax.barh(proba_df['Category'], proba_df['Probability'], color=colors)
        ax.set_xlabel('Probability')
        ax.set_title('Prediction Confidence')
        ax.set_xlim(0, 1)
        for i, (prob, cat) in enumerate(
            zip(proba_df['Probability'], proba_df['Category'])
        ):
            ax.text(prob + 0.01, i, f'{prob:.1%}', va='center')
        st.pyplot(fig)

        st.divider()

        # Disclaimer
        st.caption('''
            ⚠️ Note: This prediction is based on a Decision Tree model with 84% accuracy.
            The model tends to predict Medium for most films due to class imbalance
            in the dataset. Results should be interpreted with caution.
        ''')
