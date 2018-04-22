# CST-Dataset
Read this in other languages: [English](./README.md) [中文](./README.zh.md) 

CST-Dataset, **C**ircle-**S**quare-**T**riangle Dataset, is a simple small-scale object detection and segmentation dataset, has 1000 images, contains only circles, squares and triangles in different scales, and the file size is just 25MB. 

Considering some exists heavyweight dataset like PASCAL VOC or COCO, the training time will usually be hours or even days. So a ten-minute training dataset may serve as a good start for beginners to familiar with object detection and segmentation or as a sanity check dataset for implementors to quick check the correctness of their model implementation.

This dataset is randomly generated with some control to make sure that these objects will not overlap with each other or with border, make it ease train, some samples list as below.

## Dataset samples

![](http://chuantu.biz/t6/292/1524394809x-1404793238.png)


## Annotation format:

CSV file with colunms as below.

- [ ] TODO: COCO dataset compatible format in json file.

![](http://chuantu.biz/t6/293/1524399740x-1404758359.jpg)

## Dataset split:

|stage|samples|
|------|--------------|
|train|600|
|val|100|
|test|300|

## Dataset download link:

Check release file.

## Code to generate dataset:

Because of fixing random seed, the result of generating will be the same with release file.

```python
python3 generate_cst.py ./output_dir
```
