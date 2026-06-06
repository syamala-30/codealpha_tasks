import glob
import numpy as np
from music21 import converter, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation

# =========================
# STEP 1: LOAD MIDI FILES
# =========================
notes = []

print("Loading MIDI files...")

for file in glob.glob("data/*.mid"):
    print("Reading:", file)

    try:
        midi = converter.parse(file)

        # Faster + safer extraction
        notes_to_parse = midi.recurse().notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    except Exception as e:
        print("Skipped:", file, e)

print("Total notes extracted:", len(notes))

# Safety check
if len(notes) < 50:
    raise ValueError("Not enough notes extracted. Check MIDI files.")

# =========================
# STEP 2: OPTIMIZE DATASET
# =========================
sequence_length = 50   # reduced for speed

# optional safety cap (prevents explosion)
notes = notes[:20000]

pitchnames = sorted(set(notes))
note_to_int = {note: num for num, note in enumerate(pitchnames)}

network_input = []
network_output = []

for i in range(len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    seq_out = notes[i + sequence_length]

    network_input.append([note_to_int[n] for n in seq_in])
    network_output.append(note_to_int[seq_out])

n_patterns = len(network_input)

print("Training samples:", n_patterns)

# reshape input
X = np.reshape(network_input, (n_patterns, sequence_length, 1))
X = X / float(len(pitchnames))

y = np.eye(len(pitchnames))[network_output]

# =========================
# STEP 3: LIGHTWEIGHT MODEL
# =========================
model = Sequential()

model.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(128))
model.add(Dropout(0.2))

model.add(Dense(128))
model.add(Dense(len(pitchnames)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# =========================
# STEP 4: FAST TRAINING
# =========================
print("Training started...")

model.fit(
    X,
    y,
    epochs=5,
    batch_size=128
)

# =========================
# STEP 5: GENERATION
# =========================
print("Generating music...")

start = np.random.randint(0, len(network_input) - 1)

pattern = network_input[start]
prediction_output = []

for _ in range(200):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(len(pitchnames))

    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)

    result = pitchnames[index]
    prediction_output.append(result)

    pattern.append(index)
    pattern = pattern[1:]

# =========================
# STEP 6: SAVE MIDI
# =========================
offset = 0
output_notes = []

for pattern in prediction_output:
    if '.' in pattern:
        notes_in_chord = pattern.split('.')
        chord_notes = [note.Note(int(n)) for n in notes_in_chord]
        new_chord = chord.Chord(chord_notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        output_notes.append(new_note)

    offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='output.mid')

print("DONE: output.mid generated successfully")