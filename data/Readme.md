raw: Raw X-ray images. Pairs of two images belong together, but are flipped.

no_watermark: Contains "ground truth" images without watermark. Generated by [watermarkremove.py](http://watermarkremove.py)

[watermarkremove.py](http://watermarkremove.py): Removes watermarks by combining the flipped images in "raw" and outputting them to "no_watermark". ***no need to execute*** this script, as the images are already generated for you.