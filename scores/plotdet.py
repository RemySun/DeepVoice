import sys
import os

print(sys.argv)


bashCommand = 'plotDET.py --colour --limit 0.01 80 0.01 80 --title="DeepVoice DET" --output="det.pdf"'

for i in range(1, len(sys.argv)):
    bashCommand += ' --label="' + sys.argv[i] + '" ' + sys.argv[i]

os.system(bashCommand)
