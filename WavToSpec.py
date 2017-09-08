import matplotlib.pyplot as plt
from scipy.io import wavfile

def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 256 #length of the windowing segments.
    fs = 256  # Sampling frequency
    pxx, freq, bins, im = plt.specgram(data, nfft, fs)
    plt.axis('off')
    #plt.savefig('C:\deeplearning\Speech Stuff\LDC93S1.png',
    plt.savefig('C:\deeplearning\Speech Stuff\\test.png',
                dfi = 100, #dots per inch
                frameon = 'false',
                aspect = 'normal',
                bbox_inches = 'tight',
                pad_inches = 0) #spectrogram saved as a png file.

    
def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

if __name__ == '__main__': #main function
    wav_file = 'C:\deeplearning\Speech Stuff\LDC93S1.wav' #file name of the wav file.
    wav_file = 'C:\deeplearning\Speech Stuff\\test.wav' #file name of the wav file.
    graph_spectrogram(wav_file)
    print("Generated the spectrogram..!")
    
    
