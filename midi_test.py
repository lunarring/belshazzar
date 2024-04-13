#%%
import rtmidi
from time import sleep

# Create a new MIDI output instance
midiout = rtmidi.MidiOut()

# Open the first available MIDI output port
available_ports = midiout.get_ports()

#%%
if available_ports:
    midiout.open_port(0)
else:
    midiout.open_virtual_port("My virtual output")

try:
    note_on = [0x90, 64, 64]  # Note on channel 1, middle C, velocity 64
    note_off = [0x80, 64, 0]  # Note off channel 1, middle C, velocity 0

    for _ in range(10):  # Send note on and note off 10 times
        # Send note on
        midiout.send_message(note_on)
        sleep(1)  # Wait for 1 second
        
        # Send note off
        midiout.send_message(note_off)
        sleep(1)  # Wait for 1 second

finally:
    # Ensure the MIDI port is closed
    midiout.close_port()
    del midiout

# %%
