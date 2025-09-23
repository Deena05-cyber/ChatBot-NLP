import json
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import os

class IntelligentChatbot:
    def __init__(self, model_path='models/chatbot_model.pkl', intents_path='data/intents.json'):
        self.model_path = model_path
        self.intents_path = intents_path
        self.lemmatizer = WordNetLemmatizer()
        self.load_model()
        self.load_intents()
        self.conversation_history = []
        self.user_name = None
        
    def load_model(self):
        """Load the trained model"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.model = model_data['sklearn_model']
            self.words = model_data['words']
            self.classes = model_data['classes']
            
            print("✅ Model loaded successfully!")
            return True
            
        except FileNotFoundError:
            print("❌ Model file not found! Please train the model first.")
            print("Run: python train_model.py")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def load_intents(self):
        """Load intents for responses"""
        try:
            with open(self.intents_path, 'r') as file:
                self.intents = json.load(file)
            return True
        except FileNotFoundError:
            print(f"❌ Intents file not found: {self.intents_path}")
            return False
    
    def preprocess_input(self, sentence):
        """Preprocess user input"""
        # Tokenize the pattern - split words into array
        sentence_words = nltk.word_tokenize(sentence)
        # Stem each word - create short form for word
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return ' '.join(sentence_words)
    
    def predict_class(self, sentence, threshold=0.25):
        """Predict the intent class"""
        processed_sentence = self.preprocess_input(sentence)
        
        # Get prediction probabilities
        prediction = self.model.predict_proba([processed_sentence])
        predicted_class = self.model.predict([processed_sentence])[0]
        confidence = prediction.max()
        
        if confidence > threshold:
            return predicted_class, confidence
        else:
            return "unknown", confidence
    
    def get_response(self, intent):
        """Get response based on predicted intent"""
        list_of_intents = self.intents['intents']
        
        for i in list_of_intents:
            if i['tag'] == intent:
                result = random.choice(i['responses'])
                return result
        
        return "I'm sorry, I don't understand that. Could you please rephrase?"
    
    def extract_name(self, user_input):
        """Simple name extraction from user input"""
        user_input_lower = user_input.lower()
        
        if 'my name is' in user_input_lower:
            name_part = user_input_lower.split('my name is')[1].strip()
            name = name_part.split()[0] if name_part else None
        elif 'i am' in user_input_lower:
            name_part = user_input_lower.split('i am')[1].strip()
            name = name_part.split()[0] if name_part else None
        elif 'call me' in user_input_lower:
            name_part = user_input_lower.split('call me')[1].strip()
            name = name_part.split()[0] if name_part else None
        else:
            return None
            
        return name.capitalize() if name and name.isalpha() else None
    
    def personalize_response(self, response, user_input):
        """Add personalization to responses"""
        # Check if user is sharing their name
        if not self.user_name:
            name = self.extract_name(user_input)
            if name:
                self.user_name = name
                return f"Nice to meet you, {self.user_name}! {response}"
        
        # Add user's name to response occasionally
        if self.user_name and random.random() < 0.3:  # 30% chance
            return f"{response} {self.user_name}!"
        
        return response
    
    def chat(self, user_input):
        """Main chat function"""
        if not hasattr(self, 'model'):
            return "❌ Model not loaded. Please train the model first."
        
        # Add to conversation history
        self.conversation_history.append(f"User: {user_input}")
        
        # Predict intent
        intent, confidence = self.predict_class(user_input)
        
        # Get response
        if intent != "unknown":
            response = self.get_response(intent)
            # Personalize response
            response = self.personalize_response(response, user_input)
        else:
            # Fallback responses for unknown intents
            fallback_responses = [
                "I'm not sure I understand. Could you try rephrasing that?",
                "That's interesting! Can you tell me more?",
                "I'm still learning. Could you ask that differently?",
                "I don't have information about that. Is there something else I can help with?",
                "Hmm, I'm not certain about that. What else would you like to know?"
            ]
            response = random.choice(fallback_responses)
        
        # Add to conversation history
        self.conversation_history.append(f"Bot: {response}")
        
        # Keep only last 10 exchanges in history
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return response, confidence
    
    def get_stats(self):
        """Get chatbot statistics"""
        return {
            'total_classes': len(self.classes),
            'total_words': len(self.words),
            'classes': self.classes,
            'conversation_length': len(self.conversation_history),
            'user_name': self.user_name
        }

def main():
    """Interactive chatbot interface"""
    print("🤖 Initializing Chatbot...")
    
    # Check if model exists
    if not os.path.exists('models/chatbot_model.pkl'):
        print("❌ No trained model found!")
        print("Please run: python train_model.py first")
        return
    
    # Initialize chatbot
    bot = IntelligentChatbot()
    
    if not hasattr(bot, 'model'):
        return
    
    print("=" * 60)
    print("🎉 Chatbot Ready! Here are some things you can try:")
    print("   • Say hello or greetings")
    print("   • Ask for help")
    print("   • Say thank you")
    print("   • Ask about my name")
    print("   • Tell me your name")
    print("   • Ask about weather or time")
    print("   • Say goodbye to exit")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                print("🤖 Bot: Please say something!")
                continue
            
            # Check for exit conditions
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                stats = bot.get_stats()
                farewell = f"Goodbye{', ' + stats['user_name'] if stats['user_name'] else ''}! "
                farewell += f"We had {stats['conversation_length']//2} exchanges. Have a great day!"
                print(f"🤖 Bot: {farewell}")
                break
            
            # Get response
            response, confidence = bot.chat(user_input)
            print(f"🤖 Bot: {response}")
            
            # Show confidence for debugging (optional)
            if confidence < 0.5:
                print(f"     💡 (Low confidence: {confidence:.2f})")
                
        except KeyboardInterrupt:
            print("\n🤖 Bot: Goodbye! Thanks for chatting!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
