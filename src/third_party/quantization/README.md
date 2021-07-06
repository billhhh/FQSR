
# QTool: A low-bit quantization toolbox for deep neural networks in computer vision

This project provides aboundant choices of quantization strategies (such as the quantization algorithms, training schedules and empirical tricks) for quantizing the image classification neural networks into low-bit counterparts. Associated projects demonstrate that this project can also act as a flexible plugin and benefit other computer vision tasks, such as object detection, segmentation and text parsing. Pretrained models are provided to show high standard of the code on achieving appealing quantization performance. 

## Instructions for different tasks

- [Classification](./doc/classification.md)
- [Detection / Segmentation / Text parsing ](./doc/detectron2.md)
- [Super Resolution](./doc/edsr.md)

## Update History

- Super Resolution (in preparing)
- Text parsing (in preparing)
- 2020.07.08 Instance Segmentation
- 2020.07.08 Object Detection
- 2020.06.23 Add classification quantization

## Citation

Please cite the following work if you find the project helpful.

```
@misc{chen2020qtool,
author = {Peng Chen, Bohan Zhuang, Jing Liu and Chunlei Liu},
title = {{QTool: A low-bit quantization toolbox for deep neural networks in computer vision}},
year = {2020},
howpublished = {\url{https://github.com/blueardour/model-quantization}},
note = {Accessed: [Insert date here]}
}
```

For quantized object detection, please cite
```
@misc{liu2020aqd,
    title={AQD: Towards Accurate Quantized Object Detection},
    author={Jing Liu and Bohan Zhuang and Peng Chen and Mingkui Tan and Chunhua Shen},
    year={2020},
    eprint={2007.06919},
    archivePrefix={arXiv}
}
```

Also cite the corresponding publications when you choose [dedicated algorithms](./doc/reference.md).

## Contribute

To contribute, PR is appreciated and suggestions are welcome to discuss with. Private contact is available at blueardour@gmail.com

## License

For academic use, this project is licensed under the 2-clause BSD License. See LICENSE file.

