import imageio
import os
import glob
import re

frame_folder = 'temp_frames/'
output_gif_path = 'output.gif'


frame_files = glob.glob(os.path.join(frame_folder, "*.png"))


def extract_numbers(filename):
    numbers = re.findall(r'\d+', filename)
    return list(map(int, numbers))


frame_files = sorted(frame_files, key=extract_numbers)
frame_files_filtered = [f for f in frame_files if extract_numbers(f)[0] <= 30]

with imageio.get_writer(output_gif_path, mode='I', duration=0.1, loop=0) as writer:
    for frame_file in frame_files_filtered:
        frame = imageio.imread(frame_file)
        
        writer.append_data(frame)

print(f"GIF has been saved to {output_gif_path}")
