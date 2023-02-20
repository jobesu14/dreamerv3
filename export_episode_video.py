import imageio
import numpy as np

imageio.mimsave('/home/jonanthan/logdir/run2/video.mp4', np.load('/home/jonanthan/logdir/run2/replay/20230217T200642F563233-0PSqMt3WP9X2qzYOdp8Rj1-38cv2Pz0ToODPAUDJJoZ38-1024.npz')['image'])