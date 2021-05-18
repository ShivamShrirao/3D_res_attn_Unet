#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import numpy as np

from skimage import io
import SimpleITK as sitk

# In[ ]:


# vol = io.imread("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/attention-mri.tif")
vol = sitk.GetArrayFromImage(sitk.ReadImage("../datasets/BRATS_Dataset/brats_dataset/HGG/Brats18_2013_10_1/Brats18_2013_10_1_flair.nii.gz"))
vol = (vol/vol.max())*255
vol = vol.astype(np.uint8)
volume = vol#.T
print(volume.shape)
nb_frames, r, c = volume.shape


# In[ ]:


# Define frames
import plotly.graph_objects as go
# nb_frames = 68


# In[ ]:


fig = go.Figure(frames=[go.Frame(data=go.Surface(
    z=((nb_frames-1)/10 - k * 0.1) * np.ones((r, c)),
    surfacecolor=np.flipud(volume[nb_frames-1 - k]),
    cmin=0, cmax=255
    ),
    name=str(k) # you need to name the frame for the animation to behave properly
    )
    for k in range(nb_frames)])

# Add data to be displayed before animation starts
fig.add_trace(go.Surface(
    z=(nb_frames-1)/10 * np.ones((r, c)),
    surfacecolor=np.flipud(volume[nb_frames-1]),
    colorscale='Gray',
    cmin=0, cmax=255,
    colorbar=dict(thickness=20, ticklen=4)
    ))


def frame_args(duration):
    return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

sliders = [
            {
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], frame_args(0)],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(fig.frames)
                ],
            }
        ]

# Layout
fig.update_layout(
         title='Slices in volumetric data',
         width=600,
         height=600,
         scene=dict(
                    zaxis=dict(range=[-0.1, nb_frames/10], autorange=False),
                    aspectratio=dict(x=1, y=1, z=1),
                    ),
         updatemenus = [
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;", # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;", # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
         ],
         sliders=sliders
)


# In[ ]:


fig.show()


# In[ ]:




