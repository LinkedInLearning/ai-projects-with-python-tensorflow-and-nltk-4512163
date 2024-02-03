# Import the required library
import speech_recognition as sr

# Initialize recognizer class (for recognizing the speech)
r = sr.Recognizer()

def listen_and_transcribe(language='en', stop_phrase='stop listening'):
    # Reading Microphone as source
    # Listening the speech and store in audio_text variable
    with sr.Microphone() as source:
        # Noise cancellation
        r.adjust_for_ambient_noise(source)
        while True:  # Infinite loop to keep listening
            print("Talk")
            audio_text = r.listen(source)
            print("Time over, thanks")

            # Recognize the speech
            try:
                # Using google speech recognition
                text = r.recognize_google(audio_text, language=language)
                print("Text: "+text)

                # Stop listening if the stop_phrase is detected
                if stop_phrase in text:
                    print("Stopping as per your request.")
                    break
            except:
                print("Sorry, I did not get that")

# Call the function with the language code and stop phrase
listen_and_transcribe('en', 'stop listening')