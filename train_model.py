import json
import nltk
import numpy as np
import pickle
import random
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import os

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

class ChatbotTrainer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?', '.', ',', '!']
        
    def load_intents(self, file_path):
        """Load intents from JSON file"""
        try:
            with open(file_path, 'r') as file:
                intents = json.load(file)
            print(f"✅ Loaded intents from {file_path}")
            return intents
        except FileNotFoundError:
            print(f"❌ Error: {file_path} not found!")
            return None
        except json.JSONDecodeError:
            print(f"❌ Error: Invalid JSON in {file_path}")
            return None
    
    def preprocess_data(self, intents):
        """Preprocess the training data"""
        print("🔄 Preprocessing training data...")
        
        # Loop through each sentence in intents patterns
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                # Tokenize each word
                w = nltk.word_tokenize(pattern)
                self.words.extend(w)
                # Add documents in corpus
                self.documents.append((w, intent['tag']))
                # Add to classes list
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
        
        # Lemmatize and lower each word and remove duplicates
        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))
        
        # Sort classes
        self.classes = sorted(list(set(self.classes)))
        
        print(f"📊 Found {len(self.documents)} documents")
        print(f"📊 Found {len(self.classes)} classes: {self.classes}")
        print(f"📊 Found {len(self.words)} unique words")
        
    def create_training_data(self):
        """Create training data in the format required for machine learning"""
        print("🔄 Creating training data...")
        
        # Create training data
        training = []
        output_empty = [0] * len(self.classes)
        
        # Training set, bag of words for each sentence
        for doc in self.documents:
            # Initialize bag of words
            bag = []
            # List of tokenized words for the pattern
            pattern_words = doc[0]
            # Lemmatize each word - create base word, in attempt to represent related words
            pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in pattern_words]
            
            # Create bag of words array with 1, if word match found in current pattern
            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)
            
            # Output is a '0' for each tag and '1' for current tag (for each pattern)
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            
            training.append([bag, output_row])
        
        # Shuffle features and turn into np.array
        random.shuffle(training)
        training = np.array(training, dtype=object)
        
        # Create train and test lists. X - patterns, Y - intents
        self.train_x = list(training[:, 0])
        self.train_y = list(training[:, 1])
        
        print(f"✅ Training data created: {len(self.train_x)} samples")
    
    def train_sklearn_model(self):
        """Train sklearn model with bag of words"""
        print("🤖 Training Sklearn model...")
        
        # Prepare text data for TF-IDF
        texts = []
        labels = []
        
        for doc in self.documents:
            # Join tokenized words back to text
            text = ' '.join([self.lemmatizer.lemmatize(word.lower()) for word in doc[0]])
            texts.append(text)
            labels.append(doc[1])
        
        # Create pipeline with TF-IDF and Naive Bayes
        self.sklearn_model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.sklearn_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.sklearn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Sklearn Model trained with accuracy: {accuracy:.4f}")
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
    def save_model(self):
        """Save the trained model and preprocessed data"""
        print("💾 Saving model and data...")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save all necessary data
        model_data = {
            'sklearn_model': self.sklearn_model,
            'words': self.words,
            'classes': self.classes,
            'lemmatizer': self.lemmatizer
        }
        
        with open('models/chatbot_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("✅ Model saved successfully to models/chatbot_model.pkl")
    
    def train(self, intents_file='data/intents.json'):
        """Complete training pipeline"""
        print("🚀 Starting Chatbot Training...")
        print("=" * 50)
        
        # Load intents
        intents = self.load_intents(intents_file)
        if not intents:
            return False
        
        # Preprocess data
        self.preprocess_data(intents)
        
        # Create training data
        self.create_training_data()
        
        # Train sklearn model
        self.train_sklearn_model()
        
        # Save model
        self.save_model()
        
        print("=" * 50)
        print("🎉 Training completed successfully!")
        return True

if __name__ == "__main__":
    trainer = ChatbotTrainer()
    trainer.train()
import json
import nltk
import numpy as np
import pickle
import random
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import os

# Download required NLTK data
print("📥 Downloading NLTK data...")
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("✅ NLTK data downloaded!")
except Exception as e:
    print(f"⚠️ NLTK download warning: {e}")

class ChatbotTrainer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?', '.', ',', '!']
        
    def load_intents(self, file_path):
        """Load intents from JSON file"""
        try:
            print(f"🔍 Loading intents from: {os.path.abspath(file_path)}")
            with open(file_path, 'r', encoding='utf-8') as file:
                intents = json.load(file)
            print(f"✅ Loaded intents from {file_path}")
            print(f"📊 Found {len(intents.get('intents', []))} intent categories")
            return intents
        except FileNotFoundError:
            print(f"❌ Error: {file_path} not found!")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in {file_path}: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
    
    def preprocess_data(self, intents):
        """Preprocess the training data"""
        print("🔄 Preprocessing training data...")
        
        # Loop through each sentence in intents patterns
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                # Tokenize each word
                w = nltk.word_tokenize(pattern)
                self.words.extend(w)
                # Add documents in corpus
                self.documents.append((w, intent['tag']))
                # Add to classes list
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
        
        # Lemmatize and lower each word and remove duplicates
        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))
        
        # Sort classes
        self.classes = sorted(list(set(self.classes)))
        
        print(f"📊 Found {len(self.documents)} documents")
        print(f"📊 Found {len(self.classes)} classes: {self.classes}")
        print(f"📊 Found {len(self.words)} unique words")
        
    def create_training_data(self):
        """Create training data in the format required for machine learning"""
        print("🔄 Creating training data...")
        
        # Create training data
        training = []
        output_empty = [0] * len(self.classes)
        
        # Training set, bag of words for each sentence
        for doc in self.documents:
            # Initialize bag of words
            bag = []
            # List of tokenized words for the pattern
            pattern_words = doc[0]
            # Lemmatize each word - create base word, in attempt to represent related words
            pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in pattern_words]
            
            # Create bag of words array with 1, if word match found in current pattern
            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)
            
            # Output is a '0' for each tag and '1' for current tag (for each pattern)
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            
            training.append([bag, output_row])
        
        # Shuffle features and turn into np.array
        random.shuffle(training)
        training = np.array(training, dtype=object)
        
        # Create train and test lists. X - patterns, Y - intents
        self.train_x = list(training[:, 0])
        self.train_y = list(training[:, 1])
        
        print(f"✅ Training data created: {len(self.train_x)} samples")
    
    def train_sklearn_model(self):
        """Train sklearn model with TF-IDF"""
        print("🤖 Training Sklearn model...")
        
        # Prepare text data for TF-IDF
        texts = []
        labels = []
        
        for doc in self.documents:
            # Join tokenized words back to text
            text = ' '.join([self.lemmatizer.lemmatize(word.lower()) for word in doc[0]])
            texts.append(text)
            labels.append(doc[1])
        
        print(f"📝 Prepared {len(texts)} text samples")
        
        # Create pipeline with TF-IDF and Naive Bayes
        self.sklearn_model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        # Split data for validation
        if len(texts) > 5:
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )
        else:
            X_train, X_test, y_train, y_test = texts, texts, labels, labels
        
        # Train the model
        print(f"🔧 Training with {len(X_train)} samples...")
        self.sklearn_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.sklearn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Sklearn Model trained with accuracy: {accuracy:.4f}")
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
    def save_model(self, intents):
        """Save the trained model and preprocessed data INCLUDING INTENTS"""
        print("💾 Saving model and data...")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save all necessary data INCLUDING THE INTENTS JSON DATA
        model_data = {
            'sklearn_model': self.sklearn_model,
            'words': self.words,
            'classes': self.classes,
            'lemmatizer': self.lemmatizer,
            'intents': intents  # ← THIS WAS MISSING - NOW INCLUDED!
        }
        
        model_file = os.path.join('models', 'chatbot_model.pkl')
        
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved successfully to {os.path.abspath(model_file)}")
        
        # Verify the file was created and check intents
        if os.path.exists(model_file):
            file_size = os.path.getsize(model_file)
            print(f"📁 File size: {file_size} bytes")
            print(f"📊 Saved intents with {len(intents.get('intents', []))} categories")
            return True
        else:
            print("❌ Model file was not created!")
            return False
    
    def train(self, intents_file='data/intents.json'):
        """Complete training pipeline"""
        print("🚀 Starting Chatbot Training...")
        print("=" * 60)
        print(f"📍 Current working directory: {os.getcwd()}")
        
        # Try different possible paths for intents file
        possible_paths = [
            intents_file,
            'data/intents.json',
            '../data/intents.json',
            'intents.json'
        ]
        
        intents = None
        for path in possible_paths:
            print(f"🔍 Trying path: {path}")
            if os.path.exists(path):
                intents = self.load_intents(path)
                if intents:
                    break
        
        if not intents:
            print("❌ Could not find intents.json file!")
            print("Make sure the file exists in one of these locations:")
            for path in possible_paths:
                print(f"   - {os.path.abspath(path)}")
            return False
        
        # Preprocess data
        self.preprocess_data(intents)
        
        # Create training data
        self.create_training_data()
        
        # Train sklearn model
        self.train_sklearn_model()
        
        # Save model WITH INTENTS DATA
        if not self.save_model(intents):
            return False
        
        print("=" * 60)
        print("🎉 Training completed successfully!")
        print("✅ Intents data has been saved in the model!")
        print("✅ You can now run your web application!")
        return True

if __name__ == "__main__":
    trainer = ChatbotTrainer()
    success = trainer.train()
    
    if not success:
        print("\n🔧 Troubleshooting steps:")
        print("1. Make sure data/intents.json exists")
        print("2. Check that the JSON file has valid content")
        print("3. Ensure you have write permissions in the models folder")
        input("Press Enter to exit...")
