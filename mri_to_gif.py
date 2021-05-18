import imageio
import SimpleITK as sitk
import numpy as np

img = sitk.GetArrayFromImage(sitk.ReadImage("../datasets/BRATS_Dataset/brats_dataset/HGG/Brats18_2013_10_1/Brats18_2013_10_1_flair.nii.gz"))
img = (img/img.max())*255
img = img.astype(np.uint8)
img = np.stack((img,)*3, axis=-1)

seg = sitk.GetArrayFromImage(sitk.ReadImage("../datasets/BRATS_Dataset/brats_dataset/HGG/Brats18_2013_10_1/Brats18_2013_10_1_seg.nii.gz"))
ncr = seg == 1
ed = seg == 2
et = seg == 4

seg = (np.stack([ncr, ed, et])*255).astype(np.uint8)
seg = seg.transpose(1,2,3,0)

with imageio.get_writer('mri.gif', mode='I', fps=15) as writer:
    alpha = 0.5
    images = (img*alpha + seg*(1-alpha)).astype(np.uint8)
    for i in images:
        writer.append_data(i)