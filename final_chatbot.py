from flask import Flask, render_template, request, jsonify
import os
import pickle
import random
import json

app = Flask(__name__)

# Global variables
model_data = None
conversation_history = []
user_name = None

def load_model():
    global model_data
    try:
        # All files are in the same folder (models folder)
        model_path = 'chatbot_model.pkl'
        
        print(f"📍 Looking for model at: {os.path.abspath(model_path)}")
        
        if not os.path.exists(model_path):
            print("❌ Model file not found!")
            return False
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model keys: {list(model_data.keys())}")
        
        # Check if intents were saved in model
        if 'intents' not in model_data or not model_data['intents']:
            print("⚠️ Intents missing from model, trying to load from JSON...")
            try:
                with open('data/intents.json', 'r', encoding='utf-8') as f:
                    intents_json = json.load(f)
                model_data['intents'] = intents_json
                print("✅ Intents loaded from JSON file")
            except Exception as e:
                print(f"❌ Could not load intents from JSON: {e}")
                return False
        else:
            print(f"✅ Intents found in model: {len(model_data['intents'].get('intents', []))} categories")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_bot_response(user_input):
    global model_data, user_name
    
    if not model_data:
        return "Model not loaded!", 0.0
    
    try:
        model = model_data['sklearn_model']
        intents_data = model_data.get('intents', {})
        
        # Get prediction
        predicted_intent = model.predict([user_input.lower()])[0]
        confidence = model.predict_proba([user_input.lower()]).max()
        
        # DEBUG: Print prediction details
        print(f"🔍 DEBUG: User input: '{user_input}'")
        print(f"🔍 DEBUG: Predicted intent: '{predicted_intent}'")
        print(f"🔍 DEBUG: Confidence: {confidence:.4f}")
        
        # Check for user name
        if not user_name:
            for pattern in ['my name is', 'i am', 'call me']:
                if pattern in user_input.lower():
                    parts = user_input.lower().split(pattern)
                    if len(parts) > 1:
                        name = parts[1].strip().split()[0]
                        if name.isalpha():
                            user_name = name.capitalize()
                            break
        
        # FIXED: Proper confidence threshold and intent matching
        if confidence > 0.3:  # Lower confidence threshold for better responses
            # Get intents list from the loaded data
            intents_list = intents_data.get('intents', [])
            
            if intents_list:
                print(f"✅ DEBUG: Found {len(intents_list)} intents to search through")
                
                # Find response for predicted intent
                for intent in intents_list:
                    if intent['tag'] == predicted_intent:
                        responses = intent.get('responses', [])
                        if responses:
                            response = random.choice(responses)
                            
                            # Add personalization if name was just learned
                            if user_name and any(p in user_input.lower() for p in ['my name is', 'i am', 'call me']):
                                if '{}' in response:
                                    response = response.format(user_name)
                                else:
                                    response = f"Nice to meet you, {user_name}! {response}"
                            
                            print(f"✅ DEBUG: Found matching intent '{predicted_intent}', returning: '{response}'")
                            return response, confidence
                
                # Intent predicted but not found in data
                available_tags = [intent['tag'] for intent in intents_list]
                print(f"❌ DEBUG: Intent '{predicted_intent}' not found in available intents: {available_tags}")
                return f"I predicted '{predicted_intent}' but couldn't find responses for it. Available: {', '.join(available_tags[:5])}", confidence
            else:
                print(f"❌ DEBUG: No intents list found in intents_data")
                return "No intents data available in the model.", confidence
        else:
            # Low confidence fallback
            print(f"⚠️ DEBUG: Low confidence ({confidence:.4f}), using fallback")
            return f"I'm not sure I understand (confidence: {confidence:.1%}). Could you try rephrasing that?", confidence
        
    except Exception as e:
        print(f"❌ Error in get_bot_response: {e}")
        import traceback
        traceback.print_exc()
        return f"Sorry, I had trouble processing that: {str(e)}", 0.0

@app.route('/')
def home():
    # Since chat.html is in the same folder, we need to set up templates
    template_dir = 'templates'
    
    # Create templates folder if it doesn't exist
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
        print("✅ Created templates folder")
    
    # Move chat.html to templates if it's not there already
    if os.path.exists('chat.html') and not os.path.exists('templates/chat.html'):
        import shutil
        shutil.copy('chat.html', 'templates/chat.html')
        print("✅ Copied chat.html to templates folder")
    
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    global conversation_history
    
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message', 'status': 'error'})
        
        response, confidence = get_bot_response(user_message)
        
        # Add to conversation history
        conversation_history.append(f"User: {user_message}")
        conversation_history.append(f"Bot: {response}")
        
        # Keep only last 20 messages
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]
        
        return jsonify({
            'response': response,
            'confidence': float(confidence),
            'status': 'success',
            'user_name': user_name
        })
        
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({'error': f'Error: {str(e)}', 'status': 'error'})

@app.route('/reset')
def reset_chat():
    global conversation_history, user_name
    conversation_history = []
    user_name = None
    return jsonify({'status': 'Chat reset successfully'})

@app.route('/stats')
def stats():
    return jsonify({
        'conversation_length': len(conversation_history),
        'user_name': user_name,
        'model_loaded': model_data is not None,
        'total_classes': len(model_data.get('classes', [])) if model_data else 0,
        'intents_available': len(model_data.get('intents', {}).get('intents', [])) if model_data else 0
    })

if __name__ == '__main__':
    print("🚀 Starting Chatbot from 'models' folder...")
    print(f"📍 Current directory: {os.getcwd()}")
    print(f"📁 Files in current directory:")
    
    try:
        for file in os.listdir('.'):
            print(f"   - {file}")
    except Exception as e:
        print(f"   Error listing files: {e}")
    
    print(f"📄 chat.html exists: {os.path.exists('chat.html')}")
    print(f"📦 chatbot_model.pkl exists: {os.path.exists('chatbot_model.pkl')}")
    print(f"📁 data folder exists: {os.path.exists('data')}")
    print(f"📄 intents.json exists: {os.path.exists('data/intents.json')}")
    
    if load_model():
        print("🌐 Starting web server...")
        print("📱 Open your browser and go to: http://localhost:5000")
        print("=" * 60)
        app.run(debug=True, host='127.0.0.1', port=5000)
    else:
        print("❌ Failed to load model!")
        print("\n🔧 Try running: python train_model.py")
