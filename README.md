# SOM-and-GAN-exploration

MiniSOM references:
https://github.com/JustGlowing/minisom

https://heartbeat.fritz.ai/introduction-to-self-organizing-maps-soms-98e88b568f5d

GAN resources:
https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb#scrollTo=LLO2CxcVgS0w

https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

the som-test.py currently reads in the horse2zebra testA dataset (50 input real, 50 output of GAN fake).
The script fits 3 SOM's:
- real 
- fake
- combined

For the fake/real SOMs it compares the mappings by Jaccard similiarity (are the images that were grouped together when real still grouped together when faked as zebras?)
For the combined it checks how many pairs of real-fake images are mapped the same (is SOM able to distinguish between fake and real images?)

Variables that can be played around:
- initialization of SOM (currently picks random samples from data)
- size of SOM (currently 5X5)
- number of iterations (currently equals the number of images in set)
- other metrics of comparison of results
- other SOM training stuff (like type of neighbourhood function and size of neighbourhood)
