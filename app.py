import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved vectorizer
with open('trained_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load your trained model
model = pickle.load(open('twitter_sentiment_analysis_model.sav', 'rb'))

def predict():
    input_text = entry.get("1.0", tk.END).strip()  # Get text from the entry widget

    if input_text:
        # Transform the input text using the loaded vectorizer
        vectorized_input = vectorizer.transform([input_text])

        # Use your trained model to make predictions
        predicted = model.predict(vectorized_input)

        # Display the entered text and prediction below the button
        # Display the entered text and prediction below the button
        entered_text_label.config(text="Entered Tweet: ", font=("Arial", 12, "bold"), fg="black", wraplength=500)
        input_text_label.config(text=input_text, font=("Arial", 12), fg="black", wraplength=500)

        if predicted[0] == 0:
            prediction_label.config(text="Prediction: Negative", font=("Arial", 12, "bold"), fg="red")
        else:
            prediction_label.config(text="Prediction: Positive", font=("Arial", 12, "bold"), fg="green")
        
        # Clear the entry for new input
        entry.delete(1.0, tk.END)
    else:
        messagebox.showwarning("Warning", "Please enter a tweet for prediction.")

# Create a Tkinter window
root = tk.Tk()
root.title("Twitter Sentiment Analysis")

# Set a fixed window size
root.geometry("600x350")  # Width x Height

# Create a label and entry for text input
label = tk.Label(root, text="Enter the Tweet Below", font=("Arial", 12, "bold"))
label.pack()
entry = tk.Text(root, width=40, height=6, wrap="word")  # Set wrap to "word"
entry.pack()

# Resize the button image
button_img_path = "OrangeRoundedButton.png"  # Replace with your image file path
original_image = Image.open(button_img_path)
resized_image = original_image.resize((150, 80))  # Define your desired size here
button_img = ImageTk.PhotoImage(resized_image)

# Create a button label with the image and text
button_label = tk.Label(root, image=button_img, text="Predict", compound="center", font=("Arial", 16, "bold"), fg="white")
button_label.pack(pady=10)  # Adding padding between elements
button_label.bind("<Button-1>", lambda event: predict())  # Bind click event to predict function

# Label to display entered text and prediction
entered_text_label = tk.Label(root, text="Entered Tweet: ", font=("Arial", 12, "bold"), wraplength=500)
entered_text_label.pack()
# Create a second label for the input_text
input_text_label = tk.Label(root, text="", font=("Arial", 12), wraplength=500)
input_text_label.pack()
prediction_label = tk.Label(root, text="", font=("Arial", 12))
prediction_label.pack()

root.mainloop()