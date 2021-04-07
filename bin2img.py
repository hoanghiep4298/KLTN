import numpy as np
from PIL import Image
array = (np.random.rand(0, 2) * 256).astype(np.uint8)
print(array)
img = Image.fromarray(array)
img.save('test.png')