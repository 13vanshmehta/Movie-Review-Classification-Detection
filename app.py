import os
import pickle
import re
from typing import Dict, List, Any

import requests
import streamlit as st
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.ensemble import IsolationForest


# ---------------------------
# NLTK setup
# ---------------------------
@st.cache_resource(show_spinner=False)
def _ensure_nltk() -> None:
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)


_ensure_nltk()


# ---------------------------
# Bot Detection Model Loading
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_bot_detection_model():
    """
    Load the trained IsolationForest model for bot detection
    """
    with open('bot_detection_model.pkl', 'rb') as f:
        return pickle.load(f)


def extract_review_features(text: str) -> np.ndarray:
    """Extract linguistic features for bot detection"""
    if not isinstance(text, str):
        text = ""
    
    try:
        tokens = nltk.word_tokenize(text)
    except Exception:
        tokens = text.split()
    
    words = [t for t in tokens if t.isalpha()]
    num_chars = len(text)
    num_words = len(words)
    unique_words = len(set(w.lower() for w in words))
    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
    type_token_ratio = float(unique_words) / float(num_words) if num_words else 0.0
    
    sw = set(stopwords.words('english'))
    stopword_ratio = float(sum(1 for w in words if w.lower() in sw)) / float(num_words) if num_words else 0.0
    punctuation_ratio = float(sum(1 for ch in text if not ch.isalnum() and not ch.isspace())) / max(1, num_chars)
    uppercase_ratio = float(sum(1 for w in words if w.isupper() and len(w) > 1)) / max(1, num_words)
    digit_ratio = float(sum(1 for ch in text if ch.isdigit())) / max(1, num_chars)
    
    url_pattern = re.compile(r"https?://|www\.")
    url_count = len(url_pattern.findall(text))
    repeated_char_ratio = float(len(re.findall(r"(.)\1{2,}", text))) / max(1, num_chars)
    
    try:
        sentences = nltk.sent_tokenize(text)
        sentence_word_counts = [len(nltk.word_tokenize(s)) for s in sentences] if sentences else []
        avg_sentence_len = float(np.mean(sentence_word_counts)) if sentence_word_counts else 0.0
    except Exception:
        avg_sentence_len = 0.0
    
    return np.array([
        num_chars, num_words, avg_word_len, type_token_ratio, stopword_ratio,
        punctuation_ratio, uppercase_ratio, digit_ratio, url_count, repeated_char_ratio, avg_sentence_len
    ], dtype=float)


def detect_bot_like(review_text: str, bot_model) -> Dict[str, Any]:
    """Detect if a review is bot-like or human-like"""
    feats = extract_review_features(review_text)
    pred = bot_model.predict([feats])[0]
    score = float(bot_model.decision_function([feats])[0])
    label = 'human' if pred == 1 else 'bot-like'
    return {
        'label': label,
        'anomaly_score': score,
        'is_bot': label == 'bot-like'
    }


# ---------------------------
# Model loading and preprocessing
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    with open('movie_review_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('movie_review_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('movie_review_label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, vectorizer, label_encoder


def clean_review(text: str) -> str:
    if not isinstance(text, str):
        return ''
    stop_words = set(stopwords.words('english'))
    return ' '.join(word for word in text.split() if word.lower() not in stop_words)


def predict_sentiment(review_text: str, model, vectorizer, label_encoder) -> Dict[str, Any]:
    cleaned = clean_review(review_text)
    vec = vectorizer.transform([cleaned]).toarray()
    pred_numeric = model.predict(vec)[0]
    pred_label = label_encoder.inverse_transform([pred_numeric])[0]
    probs = model.predict_proba(vec)[0]
    prob_map = {label: float(prob) for label, prob in zip(label_encoder.classes_, probs)}
    return {
        'sentiment': pred_label,
        'probabilities': prob_map,
        'confidence': float(max(probs)),
    }


# ---------------------------
# TMDB API helpers
# ---------------------------
TMDB_BASE = 'https://api.themoviedb.org/3'
TMDB_IMG_BASE = 'https://image.tmdb.org/t/p/w500'


def _tmdb_get(path: str, api_key: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    params = params.copy() if params else {}
    params['api_key'] = api_key
    url = f"{TMDB_BASE}{path}"
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def tmdb_popular(api_key: str, page: int = 1) -> List[Dict[str, Any]]:
    data = _tmdb_get('/movie/popular', api_key, {'page': page})
    return data.get('results', [])


def tmdb_search_movies(query: str, api_key: str, page: int = 1) -> List[Dict[str, Any]]:
    if not query:
        return []
    data = _tmdb_get('/search/movie', api_key, {'query': query, 'page': page, 'include_adult': False})
    return data.get('results', [])


def tmdb_movie_details(movie_id: int, api_key: str) -> Dict[str, Any]:
    return _tmdb_get(f'/movie/{movie_id}', api_key)


def tmdb_movie_reviews(movie_id: int, api_key: str, page: int = 1) -> List[Dict[str, Any]]:
    data = _tmdb_get(f'/movie/{movie_id}/reviews', api_key, {'page': page})
    return data.get('results', [])


# ---------------------------
# UI components
# ---------------------------
def movie_card(movie: Dict[str, Any]):
    title = movie.get('title') or movie.get('name') or 'Untitled'
    overview = movie.get('overview', '') or ''
    poster_path = movie.get('poster_path')
    poster_url = f"{TMDB_IMG_BASE}{poster_path}" if poster_path else None
    movie_id = movie.get('id')
    link_url = f"?movie_id={movie_id}" if movie_id else None

    with st.container(border=True):
        cols = st.columns([1, 2])
        with cols[0]:
            if poster_url and link_url:
                st.markdown(
                    f"<a href='{link_url}' style='text-decoration:none;'>"
                    f"<img src='{poster_url}' style='width:100%; border-radius:8px;'/>"
                    f"</a>",
                    unsafe_allow_html=True,
                )
            elif poster_url:
                st.image(poster_url, width='stretch')
            else:
                st.write('No image')
        with cols[1]:
            if link_url:
                st.markdown(f"<a href='{link_url}' style='text-decoration:none;'><h3 style='margin:0'>{title}</h3></a>", unsafe_allow_html=True)
            else:
                st.subheader(title)
            short_desc = (overview[:160] + '...') if len(overview) > 160 else overview
            st.caption(short_desc)


def review_block(review_text: str, model, vectorizer, label_encoder, bot_model=None, review_number=None):
    # Add review number if provided
    if review_number:
        st.markdown(f"### 📝 REVIEW - {review_number}")
        st.divider()
    
    st.write(review_text)
    st.markdown("---")
    
    # Color-coded sentiment result
    result = predict_sentiment(review_text, model, vectorizer, label_encoder)
    sentiment = result['sentiment'].upper()
    st.markdown("### SENTIMENT ANALYSIS:")
    st.write(f"**Classification:** {sentiment}")
    st.caption(f"**Confidence:** {result['confidence']:.2%}")
    
    # Color indicators for sentiment
    if sentiment == 'POS':
        st.success("🟢 RESULT: POSITIVE - The review expresses positive sentiment")
    elif sentiment == 'NEG':
        st.error("🔴 RESULT: NEGATIVE - The review expresses negative sentiment")
    else:  # NEUTRAL
        st.info("🟡 RESULT: NEUTRAL - The review expresses neutral sentiment")
    
    with st.expander('View Probabilities'):
        for label, prob in result['probabilities'].items():
            st.write(f"- {label}: {prob:.2%}")
    
    # Add bot detection if model is available
    if bot_model:
        bot_result = detect_bot_like(review_text, bot_model)
        st.markdown("### ANOMALY DETECTION:")
        st.write(f"**Classification:** {bot_result['label'].upper()}")
        st.write(f"**Anomaly Score:** {bot_result['anomaly_score']:.4f}")
        
        if bot_result['is_bot']:
            st.error("🔴 RESULT: BOT-LIKE - This review is FAKE or generated by bots")
        else:
            st.success("🟢 RESULT: HUMAN-LIKE - This review appears AUTHENTIC")
    
    st.markdown("---")
    st.markdown("")  # Extra spacing between reviews


# ---------------------------
# Main App
# ---------------------------
st.set_page_config(page_title='Movie Sentiment Analyzer', page_icon='🎬', layout='wide')
st.title('🎬 Movie Sentiment Analyzer')

# Fixed/default API key (as requested)
api_key = os.environ.get('TMDB_API_KEY', '97e907bb9ccb43ba35d87249a2223f1c')

model, vectorizer, label_encoder = load_artifacts()

# Load bot detection model (with error handling for backward compatibility)
try:
    bot_model = load_bot_detection_model()
    bot_detection_available = True
except FileNotFoundError:
    bot_model = None
    bot_detection_available = False
    st.sidebar.info("⚠️ Bot detection model not found. Run cells 18-19 in the notebook to create it.")

tab_analyze, tab_browse = st.tabs(["Analyze Reviews", "Browse Movies"])  # Analyze first, Browse second

# Prefer query param selection if present
try:
    qp = st.query_params
    qp_val = qp.get('movie_id', None)
    if isinstance(qp_val, list):
        qp_movie_id = int(qp_val[0]) if qp_val else None
    else:
        qp_movie_id = int(qp_val) if qp_val else None
except Exception:
    qp_movie_id = None
if qp_movie_id:
    st.session_state['selected_movie_id'] = qp_movie_id


# ---------------------------
# Tab 1: Browse Movies
# ---------------------------
with tab_browse:
    st.subheader('Discover')
    qcol1, qcol2 = st.columns([3, 1])
    with qcol1:
        query = st.text_input('Search movies (leave empty to show Popular)', value='')
    with qcol2:
        page = st.number_input('Page', min_value=1, value=1)

    try:
        movies = tmdb_search_movies(query, api_key, page) if query else tmdb_popular(api_key, page)
    except Exception as e:
        st.error(f"TMDB error: {e}")
        movies = []

    if not movies:
        st.write('No movies to display.')
    else:
        # Grid of movies
        cols = st.columns(3)
        for idx, movie in enumerate(movies):
            with cols[idx % 3]:
                movie_card(movie)

    # Details and reviews panel
    selected_id = st.session_state.get('selected_movie_id')
    if selected_id:
        st.divider()
        try:
            details = tmdb_movie_details(selected_id, api_key)
        except Exception as e:
            st.error(f"Failed to load details: {e}")
            details = {}

        title = details.get('title') or details.get('name') or f'Movie {selected_id}'
        year = (details.get('release_date') or '')[:4]
        poster_path = details.get('poster_path')
        poster_url = f"{TMDB_IMG_BASE}{poster_path}" if poster_path else None

        dcols = st.columns([1, 2])
        with dcols[0]:
            if poster_url:
                st.image(poster_url, width='stretch')
        with dcols[1]:
            st.subheader(f"{title} {f'({year})' if year else ''}")
            st.write(details.get('overview') or 'No description available.')
            meta_cols = st.columns(3)
            with meta_cols[0]:
                st.metric('Rating', f"{(details.get('vote_average') or 0):.1f}")
            with meta_cols[1]:
                st.metric('Votes', f"{details.get('vote_count') or 0}")
            with meta_cols[2]:
                st.metric('Runtime', f"{details.get('runtime') or 0} min")

        st.markdown('### Reviews and Sentiment')
        try:
            reviews = tmdb_movie_reviews(selected_id, api_key)
        except Exception as e:
            st.error(f"Failed to load reviews: {e}")
            reviews = []

        if not reviews:
            st.write('No reviews found.')
        else:
            for idx, r in enumerate(reviews, 1):
                content = r.get('content') or ''
                with st.container(border=True):
                    review_block(content, model, vectorizer, label_encoder, bot_model, review_number=idx)


# ---------------------------
# Tab 2: Analyze Reviews
# ---------------------------
with tab_analyze:
    st.subheader('Search and Analyze')
    # Section A: Search by movie name and show reviews + results
    with st.container(border=True):
        st.markdown('#### Search Movie Reviews')
        s_query = st.text_input('Movie name', key='search_movie_query')
        if api_key and s_query:
            try:
                s_results = tmdb_search_movies(s_query, api_key)
            except Exception as e:
                st.error(f"TMDB error: {e}")
                s_results = []
        else:
            s_results = []

        if s_results:
            options = {f"{m.get('title') or m.get('name')} ({(m.get('release_date') or '')[:4]})": m.get('id') for m in s_results}
            choice = st.selectbox('Choose a movie', list(options.keys()))
            chosen_id = options.get(choice)
            if chosen_id:
                try:
                    s_details = tmdb_movie_details(chosen_id, api_key)
                    s_reviews = tmdb_movie_reviews(chosen_id, api_key)
                except Exception as e:
                    st.error(f"Failed to fetch details/reviews: {e}")
                    s_details, s_reviews = {}, []

                st.markdown('##### Reviews and Sentiment')
                if not s_reviews:
                    st.write('No reviews found.')
                else:
                    for idx, r in enumerate(s_reviews, 1):
                        content = r.get('content') or ''
                        with st.container(border=True):
                            review_block(content, model, vectorizer, label_encoder, bot_model, review_number=idx)
        else:
            if s_query:
                st.info('No results found for your query.')

    # Section B: User can input a review and get result
    with st.container(border=True):
        st.markdown('#### Classify Your Own Review')
        user_review = st.text_area('Enter your review text', height=160)
        if st.button('Analyze Review'):
            if user_review.strip():
                # Sentiment Analysis
                res = predict_sentiment(user_review, model, vectorizer, label_encoder)
                sentiment = res['sentiment'].upper()
                
                st.markdown("### SENTIMENT ANALYSIS:")
                st.write(f"**Classification:** {sentiment}")
                st.caption(f"**Confidence:** {res['confidence']:.2%}")
                
                # Color indicators for sentiment
                if sentiment == 'POS':
                    st.success("🟢 RESULT: POSITIVE - The review expresses positive sentiment")
                elif sentiment == 'NEG':
                    st.error("🔴 RESULT: NEGATIVE - The review expresses negative sentiment")
                else:  # NEUTRAL
                    st.info("🟡 RESULT: NEUTRAL - The review expresses neutral sentiment")
                
                with st.expander('View Probabilities'):
                    for label, prob in res['probabilities'].items():
                        st.write(f"- {label}: {prob:.2%}")
                
                # Bot Detection (if available)
                if bot_model:
                    st.divider()
                    bot_result = detect_bot_like(user_review, bot_model)
                    st.markdown("### AUTHENTICITY CHECK:")
                    st.write(f"**Classification:** {bot_result['label'].upper()}")
                    st.write(f"**Anomaly Score:** {bot_result['anomaly_score']:.4f}")
                    
                    if bot_result['is_bot']:
                        st.error("🔴 RESULT: BOT-LIKE - This review is FAKE or generated by bots")
                    else:
                        st.success("🟢 RESULT: HUMAN-LIKE - This review appears AUTHENTIC")
            else:
                st.warning('Please enter some text to analyze.')