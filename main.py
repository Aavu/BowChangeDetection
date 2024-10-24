from bow_change_detection import detect_bow_changes
import matplotlib.pyplot as plt


if __name__ == "__main__":
    e = detect_bow_changes(audio_path="/Users/violinsimma/PyCharmProjects/AutoAlapana/violin16k.wav", fs=16000, hop=160, threshold=0.03, dev="cpu", min_bow_length_ms=250)
    plt.plot(e)
    plt.show()