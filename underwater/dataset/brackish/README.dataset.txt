# Brackish Underwater > 1920x1080
https://universe.roboflow.com/brad-dwyer/brackish-underwater

Provided by [Aalborg University](https://vbn.aau.dk/en/projects/marine-analytics-using-computer-vision)
License: CC BY 4.0

![Example image from the dataset](https://i.imgur.com/3dtuNhv.png)

## Dataset Information

This dataset contains 14,674 images (12,444 of which contain objects of interest with bounding box annotations) of fish, crabs, and other marine animals. It was collected with a camera mounted 9 meters below the surface on the Limfjords bridge in northern Denmark by Aalborg University.

### Composition

Roboflow has extracted and processed the frames from the source videos and converted the annotations for use with many popular computer vision models. We have maintained the same 80/10/10 [train/valid/test split](https://blog.roboflow.com/train-test-split/) as the original dataset.

The class balance in the annotations is as follows:
![Class Balance](https://i.imgur.com/3MUk7D7.png)

Most of the identified objects are congregated towards the bottom of the frames.

![Annotation Heatmap](https://i.imgur.com/jAbb2i4.png)

### More Information

For more information, see the [Detection of Marine Animals in a New Underwater Dataset with Varying Visibility](https://openaccess.thecvf.com/content_CVPRW_2019/papers/AAMVEM/Pedersen_Detection_of_Marine_Animals_in_a_New_Underwater_Dataset_with_CVPRW_2019_paper.pdf) paper.

If you find the dataset useful, the authors request that you please cite their paper:

```
@InProceedings{pedersen2019brackish,
    title={Detection of Marine Animals in a New Underwater Dataset with Varying Visibility},
    author={Pedersen, Malte and Haurum, Joakim Bruslund and Gade, Rikke and Moeslund, Thomas B. and Madsen, Niels},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month = {June},
    year = {2019}
}
```