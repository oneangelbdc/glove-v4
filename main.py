import os
import sys
import subprocess
from collections import Counter
import re
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
import psutil

import nltk

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity

# Memory tracking functions
def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    bytes_used = memory_info.rss  # Resident Set Size
    
    # Convert to MB and GB
    mb_used = bytes_used / (1024 * 1024)
    gb_used = bytes_used / (1024 * 1024 * 1024)
    
    # Get percentage of total system memory
    total_memory = psutil.virtual_memory().total
    percentage = (bytes_used / total_memory) * 100
    
    return percentage, mb_used, gb_used

def format_memory_metrics():
    """Format memory usage for display"""
    percentage, mb_used, gb_used = get_memory_usage()
    return (f"Memory Usage:\n"
            f"  Percentage: {percentage:.2f}%\n"
            f"  Absolute: {mb_used:.2f} MB ({gb_used:.2f} GB)")

# Load GloVe word vectors
def load_glove_model(glove_file):
    print("Loading Glove Model...")
    model = {}
    try:
        with open(glove_file, "r", encoding="utf-8") as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array([float(val) for val in split_line[1:]])
                model[word] = embedding
        print(f"Done. {len(model)} words loaded!")
        return model
    except FileNotFoundError:
        error_msg = f"Error: GloVe file '{glove_file}' not found.\nPlease download GloVe embeddings and place them in the current directory."
        print(error_msg)
        messagebox.showerror("Error", error_msg)
        return {}

glove_model = load_glove_model("glove.6B.100d.txt")
if not glove_model:
    sys.exit(1)

# ==================== BALANCED DOMAIN DETECTION ====================
def detect_document_type_with_details(article_text):
    """
    Balanced domain detection with comprehensive term coverage for all domains
    """
    sentences = nltk.sent_tokenize(article_text)
    words = nltk.word_tokenize(article_text.lower())
    
    # Comprehensive academic indicators
    academic_terms = [
        # Research methodology
        'methodology', 'hypothesis', 'experiment', 'results', 'conclusion', 
        'analysis', 'data', 'sample', 'participants', 'procedure',
        
        # Academic structure
        'abstract', 'introduction', 'references', 'appendix', 'literature review',
        'background', 'discussion', 'limitations', 'future work',
        
        # Academic verbs
        'investigate', 'examine', 'analyze', 'demonstrate', 'conclude',
        'suggest', 'indicate', 'observe', 'measure', 'calculate',
        
        # Citations and references
        'et al', 'reference', 'cited', 'previous work', 'prior research',
        'according to', 'studies have shown', 'research indicates',
        
        # Academic qualifiers
        'significant', 'correlation', 'probability', 'variance', 'deviation'
    ]
    
    # Comprehensive legal indicators
    legal_terms = [
        # Legal document structure
        'whereas', 'hereby', 'hereinafter', 'pursuant', 'party', 
        'agreement', 'section', 'article', 'clause', 'provision',
        
        # Legal roles
        'plaintiff', 'defendant', 'appellant', 'respondent', 'petitioner',
        'lessor', 'lessee', 'licensor', 'licensee', 'guarantor',
        
        # Legal actions
        'shall', 'must', 'may', 'entitled', 'obligated', 'liable',
        'warranty', 'indemnify', 'arbitration', 'jurisdiction',
        
        # Court and legal system
        'court', 'judge', 'testimony', 'evidence', 'exhibit',
        'statute', 'regulation', 'ordinance', 'code', 'act'
    ]
    
    # Comprehensive news indicators
    news_terms = [
        # News organizations and formats
        'cnn', 'reuters', 'associated press', 'ap', 'bbc', 'fox news',
        'reported', 'announced', 'confirmed', 'according to sources',
        
        # Time references
        'today', 'yesterday', 'this week', 'recently', 'latest',
        'friday', 'monday', 'earlier', 'later', 'update',
        
        # News locations and attribution
        'officials said', 'spokesperson', 'authorities', 'police',
        'government', 'federal', 'local', 'witnesses', 'residents',
        
        # News verbs and phrases
        'said', 'told', 'stated', 'added', 'explained', 'commented',
        'breaking', 'exclusive', 'developing', 'ongoing'
    ]
    
    # Feature extraction
    features = {
        # Academic features
        'academic_terms': 0,
        'citation_patterns': len(re.findall(r'\[\d+\]|\([A-Za-z]+,?\s*\d{4}\)|et al\.', article_text)),
        'formal_structure': len(re.findall(r'\n(Abstract|Introduction|Methodology|Results|Conclusion|References)\n', article_text, re.IGNORECASE)),
        'equation_density': len(re.findall(r'@xmath|equation|fig\.|table', article_text)),
        
        # Legal features
        'legal_terms': 0,
        'contract_language': len(re.findall(r'\b(shall|must|obligated|liable|warranty)\b', article_text, re.IGNORECASE)),
        'legal_references': len(re.findall(r'\b(court|judge|statute|regulation|jurisdiction)\b', article_text, re.IGNORECASE)),
        
        # News features
        'news_terms': 0,
        'quote_density': len(re.findall(r'["\']', article_text)) / len(sentences) if sentences else 0,
        'time_references': len(re.findall(r'\b(today|yesterday|this week|recently|latest)\b', article_text, re.IGNORECASE)),
        'news_organizations': len(re.findall(r'\b(cnn|reuters|associated press|ap|bbc|fox news)\b', article_text, re.IGNORECASE)),
        
        # Structural features (neutral)
        'sentence_length_var': np.var([len(nltk.word_tokenize(s)) for s in sentences]) if len(sentences) > 1 else 0,
        'avg_sentence_length': np.mean([len(nltk.word_tokenize(s)) for s in sentences]) if sentences else 0,
        'vocabulary_richness': len(set(words)) / len(words) if words else 0
    }
    
    # Count terms for each domain
    for term_list, feature_key in [(academic_terms, 'academic_terms'), 
                                   (legal_terms, 'legal_terms'), 
                                   (news_terms, 'news_terms')]:
        count = 0
        for term in term_list:
            pattern = r'\b' + re.escape(term) + r'\b'
            count += len(re.findall(pattern, article_text, re.IGNORECASE))
        features[feature_key] = count
    
    # Detailed term tracking
    detected_terms = {
        'academic_terms_found': [],
        'legal_terms_found': [],
        'news_terms_found': [],
        'citations_found': re.findall(r'\[\d+\]|\([A-Za-z]+,?\s*\d{4}\)', article_text),
        'structural_features': []
    }
    
    # Find specific terms for each domain
    for term_list, category in [(academic_terms, 'academic_terms_found'),
                                (legal_terms, 'legal_terms_found'),
                                (news_terms, 'news_terms_found')]:
        for term in term_list:
            pattern = r'\b' + re.escape(term) + r'\b'
            found = re.findall(pattern, article_text, re.IGNORECASE)
            if found:
                detected_terms[category].extend(found)
    
    # Remove duplicates
    unique_terms = {}
    for category, terms in detected_terms.items():
        unique_terms[category] = list(set(terms))
    
    # Add structural features to display
    unique_terms['structural_features'] = [
        f"sentence_var:{features['sentence_length_var']:.1f}",
        f"avg_sent_len:{features['avg_sentence_length']:.1f}",
        f"vocab_richness:{features['vocabulary_richness']:.3f}"
    ]
    
    # BALANCED scoring system - equal weighting for all domains
    scores = {
        'academic': (
            features['academic_terms'] * 2 +
            features['citation_patterns'] * 3 +
            features['formal_structure'] * 4 +
            features['equation_density'] * 2 +
            (1 if features['vocabulary_richness'] > 0.4 else 0) * 2
        ),
        
        'legal': (
            features['legal_terms'] * 2 +
            features['contract_language'] * 3 +
            features['legal_references'] * 4 +
            (1 if features['sentence_length_var'] > 80 else 0) * 2 +
            (1 if features['avg_sentence_length'] > 25 else 0) * 2
        ),
        
        'news': (
            features['news_terms'] * 2 +
            features['quote_density'] * 8 +  # Quotes are very news-specific
            features['time_references'] * 3 +
            features['news_organizations'] * 6 +  # News orgs are definitive
            (1 if features['sentence_length_var'] < 40 else 0) * 2
        )
    }
    
    # Determine dominant type
    max_score = max(scores.values())
    if max_score > 3:  # Lower threshold for more sensitivity
        for doc_type, score in scores.items():
            if score == max_score:
                return doc_type, features, unique_terms, scores
    
    return 'general', features, unique_terms, scores

def get_domain_optimized_parameters(doc_type, article_text):
    """
    Return optimized MMR parameters for each document type
    """
    sentences = nltk.sent_tokenize(article_text)
    base_params = {
        'academic': {
            'alpha': 0.2,      # Lower positional weight (academic structure varies)
            'lambda_param': 0.8, # Higher diversity penalty (avoid redundant concepts)
            'target_ratio': max(0.15, min(0.25, 20/len(sentences))) if sentences else 0.2
        },
        'legal': {
            'alpha': 0.1,      # Very low positional weight (legal clauses equally important)
            'lambda_param': 0.9, # Very high diversity penalty (each clause unique)
            'target_ratio': max(0.1, min(0.2, 15/len(sentences))) if sentences else 0.15
        },
        'news': {
            'alpha': 0.4,      # Higher positional weight (inverted pyramid)
            'lambda_param': 0.7, # Moderate diversity penalty
            'target_ratio': max(0.2, min(0.4, 25/len(sentences))) if sentences else 0.3
        },
        'general': {
            'alpha': 0.3,      # Balanced approach
            'lambda_param': 0.7,
            'target_ratio': max(0.15, min(0.3, 20/len(sentences))) if sentences else 0.25
        }
    }
    return base_params.get(doc_type, base_params['general'])

# ==================== DOMAIN-AWARE MMR SUMMARIZATION ====================
def domain_aware_mmr_summarization(article_text, glove_model):
    """
    Enhanced MMR with automatic domain detection and optimized parameters
    """
    if not article_text:
        messagebox.showerror("Error", "No article text provided for summarization.")
        return "Unable to generate summary: Missing input data.", 'error', {}, {}, {}
    
    if not glove_model:
        messagebox.showerror("Error", "GloVe model not loaded.")
        return "Unable to generate summary: GloVe model not available.", 'error', {}, {}, {}
    
    # Detect document type with details
    doc_type, features, unique_terms, scores = detect_document_type_with_details(article_text)
    params = get_domain_optimized_parameters(doc_type, article_text)
    
    print(f"Detected document type: {doc_type}")
    print(f"Using parameters - alpha: {params['alpha']}, lambda: {params['lambda_param']}, ratio: {params['target_ratio']}")
    
    sentences = nltk.sent_tokenize(article_text)
    total_sentences = len(sentences)
    
    if total_sentences == 0:
        return "", doc_type, features, unique_terms, scores
    
    # Calculate sentence embeddings
    sentence_embeddings = []
    valid_sentences = []
    
    for i, sentence in enumerate(sentences):
        words = nltk.word_tokenize(sentence.lower())
        first_embedding = next(iter(glove_model.values()))
        sentence_embedding = np.zeros_like(first_embedding)
        word_count = 0
        
        for word in words:
            if word in glove_model:
                sentence_embedding += glove_model[word]
                word_count += 1
        
        if word_count > 0:
            sentence_embedding /= word_count
            sentence_embeddings.append(sentence_embedding)
            valid_sentences.append((i, sentence))
    
    if not sentence_embeddings:
        return sentences[0] if sentences else "No valid sentences found.", doc_type, features, unique_terms, scores
    
    sentence_embeddings = np.array(sentence_embeddings)
    
    # Calculate document centroid
    doc_centroid = np.mean(sentence_embeddings, axis=0)
    doc_centroid = doc_centroid.reshape(1, -1)
    
    # Relevance scoring
    relevance_scores = cosine_similarity(sentence_embeddings, doc_centroid).flatten()
    
    # MMR selection process
    summary_sentences = []
    summary_indices = []
    summary_embeddings = []
    
    num_summary_sentences = max(1, int(len(valid_sentences) * params['target_ratio']))
    print(f"Target summary sentences: {num_summary_sentences}")
    
    # Select first sentence (usually important)
    if valid_sentences:
        first_idx = 0
        summary_sentences.append(valid_sentences[first_idx][1])
        summary_indices.append(valid_sentences[first_idx][0])
        summary_embeddings.append(sentence_embeddings[first_idx])
    
    # MMR for remaining sentences
    remaining_indices = [i for i in range(len(valid_sentences)) if i != first_idx]
    
    while len(summary_sentences) < num_summary_sentences and remaining_indices:
        best_score = -float('inf')
        best_idx = None
        
        for idx in remaining_indices:
            sent_idx, sentence = valid_sentences[idx]
            sent_embedding = sentence_embeddings[idx].reshape(1, -1)
            
            relevance = relevance_scores[idx]
            
            # Diversity calculation
            max_similarity = 0
            if summary_embeddings:
                summary_matrix = np.array(summary_embeddings)
                similarities = cosine_similarity(sent_embedding, summary_matrix)
                max_similarity = np.max(similarities)
            
            # MMR scoring
            mmr_score = relevance - (max_similarity * params['lambda_param'])
            
            # Positional weighting
            pos_score = positional_weight(sent_idx, total_sentences)
            
            # Final score
            final_score = params['alpha'] * pos_score + (1 - params['alpha']) * mmr_score
            
            if final_score > best_score:
                best_score = final_score
                best_idx = idx
        
        if best_idx is not None:
            sent_idx, sentence = valid_sentences[best_idx]
            summary_sentences.append(sentence)
            summary_indices.append(sent_idx)
            summary_embeddings.append(sentence_embeddings[best_idx])
            remaining_indices.remove(best_idx)
    
    # Sort by original order
    sorted_summary = [sentence for _, sentence in sorted(zip(summary_indices, summary_sentences))]
    
    result = " ".join(sorted_summary)
    print(f"Generated summary: {len(nltk.sent_tokenize(result))} sentences")
    return result, doc_type, features, unique_terms, scores

def positional_weight(sentence_index, total_sentences):
    """Positional weighting for sentence importance"""
    normalized_pos = sentence_index / total_sentences
    if normalized_pos < 0.1 or normalized_pos > 0.9:
        return 0.8
    elif normalized_pos < 0.2 or normalized_pos > 0.8:
        return 0.5
    else:
        return 0.3

# ==================== SCROLLABLE TEXT FRAME CLASS ====================
class ScrollableTextFrame:
    def __init__(self, parent, title, height=6):
        self.frame = Frame(parent)
        self.frame.pack(fill="x", pady=5)
        
        # Title
        Label(self.frame, text=title, font=("Arial", 11, "bold")).pack(anchor="w")
        
        # Text area with scrollbar
        text_frame = Frame(self.frame)
        text_frame.pack(fill="x", pady=5)
        
        self.text = Text(text_frame, wrap=WORD, height=height, width=100)
        self.scrollbar = Scrollbar(text_frame, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=self.scrollbar.set)
        
        self.text.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

# ==================== GUI CLASS ====================
class SummarizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Text Summarization: Domain-Aware MMR")

        # Create main frame with scrollbar
        main_container = Frame(root)
        main_container.pack(fill="both", expand=True)
        
        # Create canvas and scrollbar for main container
        self.canvas = Canvas(main_container)
        self.scrollbar = Scrollbar(main_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        
        # Initialize variables
        self.df = None
        self.articles = []
        self.csv_summaries = []
        
        self.setup_interface()

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def setup_interface(self):
        """Setup the interface with comparison"""
        # Header
        Label(self.scrollable_frame, text="Enhanced Text Summarization System", 
              font=("Arial", 14, "bold")).pack(pady=10)
        Label(self.scrollable_frame, text="Domain-Aware Document Summarization", 
              font=("Arial", 11)).pack(pady=5)

        # Controls Frame
        control_frame = Frame(self.scrollable_frame)
        control_frame.pack(pady=10, fill="x")
        
        # File loading
        self.btnOpenFile = Button(control_frame, text="Load CSV Dataset", 
                                 command=self.open_csv, width=20)
        self.btnOpenFile.pack(pady=5)

        # Article selection
        self.selected_article = StringVar()
        self.selected_article.set("Select Article")
        self.article_dropdown = OptionMenu(control_frame, self.selected_article, "Select Article")
        self.article_dropdown.pack(pady=5)

        # Generate button
        self.btnSummarize = Button(control_frame, text="Generate Domain-Aware Summary", 
                                  command=self.evaluate, bg="lightgreen", width=25)
        self.btnSummarize.pack(pady=10)

        # Results area
        self.create_results_area()

    def create_results_area(self):
        """Create the results area with scrollable text frames"""
        # Original Article
        self.article_frame = ScrollableTextFrame(self.scrollable_frame, "Original Article:", height=6)
        
        # Document Analysis
        self.analysis_label = Label(self.scrollable_frame, text="Document Analysis: Not analyzed", 
                                  justify=LEFT, font=("Arial", 9), wraplength=1200)
        self.analysis_label.pack(anchor="w", pady=5)
        
        # Reference Summary
        self.ref_frame = ScrollableTextFrame(self.scrollable_frame, "Reference Summary:", height=4)
        
        # Domain-Aware Summary
        self.domain_frame = ScrollableTextFrame(self.scrollable_frame, "Domain-Aware Summary:", height=4)
        # Change title color for domain-aware
        self.domain_frame.frame.winfo_children()[0].config(fg="darkgreen")
        
        # Results
        Label(self.scrollable_frame, text="Results:", 
              font=("Arial", 11, "bold")).pack(anchor="w", pady=(10,5))
        
        self.results_frame = Frame(self.scrollable_frame)
        self.results_frame.pack(fill="x", pady=5)
        
        # Domain-aware results
        self.domain_results = Label(self.results_frame, text="Domain-Aware: Not evaluated", 
                                    justify=LEFT, font=("Arial", 9), fg="darkgreen")
        self.domain_results.pack(anchor="w")
        
        # Memory Usage Results (only shown after summarization)
        self.memory_label = Label(self.results_frame, text="", 
                                 justify=LEFT, font=("Arial", 9), fg="purple")
        # Don't pack it yet - only show after summarization
        
        # Performance summary
        self.performance_summary = Label(self.results_frame, text="", 
                                      justify=LEFT, font=("Arial", 10, "bold"))
        self.performance_summary.pack(anchor="w", pady=(5,0))

    def open_csv(self):
        """Load CSV dataset"""
        self.filename = filedialog.askopenfilename(
            initialdir="/", title="Select CSV File",
            filetypes=(("CSV files", "*.csv"), ("all files", "*.*"))
        )
        if self.filename:
            try:
                self.df = pd.read_csv(self.filename)
                if "article" not in self.df.columns or "summary" not in self.df.columns:
                    messagebox.showerror("Error", "CSV file must contain 'article' and 'summary' columns")
                    return
                self.articles = self.df["article"].tolist()
                self.csv_summaries = self.df["summary"].tolist()
                
                # Update dropdown
                self.article_dropdown['menu'].delete(0, 'end')
                for i in range(len(self.articles)):
                    self.article_dropdown['menu'].add_command(
                        label=f"Article {i+1}", 
                        command=lambda value=i: self.selected_article.set(f"Article {value+1}")
                    )
                
                print(f"Loaded {len(self.articles)} articles from CSV")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV file: {e}")

    def evaluate(self):
        """Handle summarization and evaluation"""
        if not self.articles:
            messagebox.showerror("Error", "Please load a CSV file first")
            return
            
        if self.selected_article.get() == "Select Article":
            messagebox.showerror("Error", "Please select an article")
            return

        selected_index = int(self.selected_article.get().split()[1]) - 1
        if selected_index >= len(self.articles):
            messagebox.showerror("Error", "Invalid article selection")
            return

        selected_article = self.articles[selected_index]
        csv_summary = self.csv_summaries[selected_index]

        # Generate domain-aware summary with detailed analysis
        domain_summary, doc_type, features, unique_terms, scores = domain_aware_mmr_summarization(selected_article, glove_model)

        # Update document analysis display with detailed terms
        analysis_text = self.format_analysis_text(doc_type, features, unique_terms, scores, selected_article)
        self.analysis_label.config(text=analysis_text)

        # Update memory usage display
        self.memory_label.config(text=format_memory_metrics())
        self.memory_label.pack(anchor="w", pady=(10,0))

        # Update display
        self.update_results(selected_article, domain_summary, csv_summary, doc_type)

    def format_analysis_text(self, doc_type, features, unique_terms, scores, article_text):
        """Enhanced analysis display with detected terms"""
        sentences = nltk.sent_tokenize(article_text)
        words = nltk.word_tokenize(article_text.lower())
        unique_ratio = len(set(words)) / len(words) if words else 0
        
        # Build the analysis text
        analysis = (f"Document Analysis:\n"
                   f"  • Type: {doc_type.upper()} (Score: {scores.get(doc_type, 0):.1f})\n"
                   f"  • Sentences: {len(sentences)}, Words: {len(words)}, Unique Ratio: {unique_ratio:.3f}\n"
                   f"  • Key Features:\n"
                   f"     Academic: terms={features['academic_terms']}, citations={features['citation_patterns']}, equations={features['equation_density']}\n"
                   f"     Legal: terms={features['legal_terms']}, contract_lang={features['contract_language']}, legal_refs={features['legal_references']}\n"
                   f"     News: terms={features['news_terms']}, quotes={features['quote_density']:.2f}, time_refs={features['time_references']}\n")
        
        # Add detected terms (only if any were found)
        analysis += "  • Detected Terms:\n"
        
        term_categories = [
            ("Academic", unique_terms.get('academic_terms_found', [])),
            ("Legal", unique_terms.get('legal_terms_found', [])),
            ("News", unique_terms.get('news_terms_found', [])),
            ("Citations", unique_terms.get('citations_found', [])),
            ("Structure", unique_terms.get('structural_features', []))
        ]
        
        terms_found = False
        for category, terms in term_categories:
            if terms and category != "Structure":  # Handle structure separately
                terms_found = True
                # Limit display to first 3 terms to avoid clutter
                display_terms = terms[:3]
                remaining = len(terms) - 3
                term_text = ", ".join(display_terms)
                if remaining > 0:
                    term_text += f" (+{remaining} more)"
                analysis += f"     {category}: {term_text}\n"
            elif category == "Structure" and terms:
                analysis += f"     {category}: {', '.join(terms)}\n"
        
        if not terms_found and 'structural_features' not in unique_terms:
            analysis += "     No specific domain terms detected\n"
        
        return analysis

    def update_results(self, article, domain_summary, csv_summary, doc_type):
        """Update the results display"""
        # Clear existing text
        self.article_frame.text.delete(1.0, END)
        self.ref_frame.text.delete(1.0, END)
        self.domain_frame.text.delete(1.0, END)
        
        # Insert new content
        self.article_frame.text.insert(END, article)
        self.ref_frame.text.insert(END, csv_summary)
        self.domain_frame.text.insert(END, domain_summary)

        # ROUGE Evaluation
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            
            domain_scores = scorer.score(csv_summary, domain_summary)
            
            # Update domain-aware results
            domain_text = (f"Domain-Aware ({doc_type.upper()}) - "
                          f"ROUGE-1: {domain_scores['rouge1'].fmeasure:.4f} | "
                          f"ROUGE-L: {domain_scores['rougeL'].fmeasure:.4f} | "
                          f"Sentences: {len(nltk.sent_tokenize(domain_summary))}")
            self.domain_results.config(text=domain_text)
            
            # Update memory usage
            self.memory_label.config(text=format_memory_metrics())
            
            # Performance summary
            avg_score = (domain_scores['rouge1'].fmeasure + domain_scores['rougeL'].fmeasure) / 2
            performance = f"Domain-aware summarization achieved average ROUGE score of {avg_score:.4f} for {doc_type} document"
            color = "darkgreen" if avg_score > 0.3 else "darkred" if avg_score < 0.2 else "black"
                
            self.performance_summary.config(text=performance, fg=color)
            
        except Exception as e:
            error_msg = f"ROUGE evaluation error: {e}"
            print(error_msg)
            self.domain_results.config(text="Domain-Aware: Evaluation failed")
            self.performance_summary.config(text="Performance evaluation unavailable")

# Main application
if __name__ == "__main__":
    print("Enhanced Text Summarization System Starting...")
    print("Features: Domain-Aware Document Summarization")
    print("=" * 60)
    
    root = Tk()
    root.geometry("1400x900")
    root.title("Research: Domain-Aware Summarization Enhancement")
    gui = SummarizationGUI(root)
    root.mainloop()