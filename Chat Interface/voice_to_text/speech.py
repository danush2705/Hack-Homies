import speech_recognition as sr
from googletrans import Translator

def transcribe_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Please speak in Tamil:")
        audio = recognizer.listen(source)

    try:
        print("Transcribing...")
        text = recognizer.recognize_google(audio, language="ta-IN")
        return text
    except sr.UnknownValueError:
        print("Sorry, could not understand the audio.")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

def translate_text(text):
    translator = Translator()
    translated_text = translator.translate(text, src='ta', dest='en').text
    return translated_text

def main():
    print("Tamil to English Speech-to-Text and Translation")
    translated_text = transcribe_speech()
    print("You (Tamil):", translated_text)
    print("Translating to English...")
    translated_text_english = translate_text(translated_text)
    print("You (English):", translated_text_english)

if __name__ == "__main__":
    main()
