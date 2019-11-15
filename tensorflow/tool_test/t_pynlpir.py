
import pynlpir


s="他说的确实在理"

pynlpir.open()
segments=pynlpir.segment(s,pos_names='all',pos_english=False)
for segment in segments:
    print(segment[0],"     ",segment[1])

key_words = pynlpir.get_key_words(s, weighted=True)
for key_word in key_words:
    print(key_word[0], '\t', key_word[1])
