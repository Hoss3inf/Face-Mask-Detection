<h1 align="center">Face Mask Detection</h1>

<div align= "center">
  <h4>Face Mask Detection system built with OpenCV, Keras/TensorFlow using Deep Learning and Computer Vision concepts in order to detect face masks in static images as well as in real-time video streams.</h4>
</div>

## :warning: TechStack/framework used

- [OpenCV](https://opencv.org/)
- [Caffe-based face detector](https://caffe.berkeleyvision.org/)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)

## :star: Features
Our face mask detector didn't use any morphed masked images dataset. The model is accurate, and since we used the MobileNetV2 architecture, it’s also computationally efficient and thus making it easier to deploy the model to embedded systems (Raspberry Pi, Google Coral, etc.).

This system can therefore be used in real-time applications which require face-mask detection for safety purposes due to the outbreak of Covid-19. This project can be integrated with embedded systems for application in airports, railway stations, offices, schools, and public places to ensure that public safety guidelines are followed.

## :file_folder: Dataset
The dataset used can be downloaded here - [Click to Download](https://drive.google.com/drive/folders/1XDte2DL2Mf_hw4NsmGst7QtYoU7sMBVG?usp=sharing)

This dataset consists of __3835 images__ belonging to two classes:
*	__with_mask: 1916 images__
*	__without_mask: 1919 images__

The images used were real images of faces wearing masks. The images were collected from the following sources:

* __Bing Search API__ ([See Python script](https://github.com/chandrikadeb7/Face-Mask-Detection/blob/master/search.py))
* __Kaggle datasets__ 
* __RMFD dataset__ ([See here](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset))

## :key: Prerequisites

All the dependencies and required libraries are included in the file <code>requirements.txt</code> [See here](https://github.com/chandrikadeb7/Face-Mask-Detection/blob/master/requirements.txt)

## 🚀&nbsp; نصب
1. ریپوزیتوری را کلون کنید
```
$ git clone https://github.com/Hoss3inf/Face-Mask-Detection.git
```

2. به پوشه Face-Mask_Detection وارد شود
```
$ cd Face-Mask-Detection/
```

3. حالا کد زیر را اجرا کنید تا کتابخانه های مورد نیاز نصب شود
```
$ pip3 install -r requirements.txt
```

## :bulb: اجرا

1. کامند پرامپت را باز کنید و دستور زیر را اجرا کنید
```
$ python3 train_mask_detector.py --dataset dataset
```

2. برای تشخیص از طریق عکسی که داریم کد زیر را اجرا می کنیم
```
$ python3 detect_mask_image.py --image images/pic1.jpeg
```

3. برای تشخیص عکس از طریق وب کم به صورت زنده کد زیر را اجرا می کنیم
```
$ python3 detect_mask_video.py 
```
## :key: Results

#### Our model gave 93% accuracy for Face Mask Detection after training via <code>tensorflow-gpu==2.0.0</code>

![](https://github.com/chandrikadeb7/Face-Mask-Detection/blob/master/Readme_images/Screenshot%202020-06-01%20at%209.48.27%20PM.png)

#### We got the following accuracy/loss training curve plot
![](https://github.com/chandrikadeb7/Face-Mask-Detection/blob/master/plot.png)

## Streamlit app

Face Mask Detector webapp using Tensorflow & Streamlit

command
```
$ streamlit run app.py 
```
## Images

<p align="center">
  <img src="Readme_images/1.PNG">
</p>
<p align="center">Upload Images</p>

<p align="center">
  <img src="Readme_images/2.PNG">
</p>
<p align="center">Results</p>

## :clap: And it's done!
Feel free to mail me for any doubts/query 
:email: chandrikadeb7@gmail.com

## :handshake: Contribution
Feel free to **file a new issue** with a respective title and description on the the [Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection/issues) repository. If you already found a solution to your problem, **I would love to review your pull request**! 

## :trophy: Awards
Awarded Runners Up position in [Amdocs Innovation India ICE Project Fair]( https://www.amdocs.com/)

![](Readme_images/nn.jpeg)

## 👏 Appreciation

Selected in [Devscript Winter Of Code](https://devscript.tech/woc/)

![](Readme_images/Devscript.jpeg)

Selected in [Script Winter Of Code](https://swoc.tech/project.html)

![](Readme_images/winter.jpeg)

Seleted in [Student Code-in](https://scodein.tech/)

![](Readme_images/sci.jpeg)

## :heart: Owner
Made with :heart:&nbsp;  by [Chandrika Deb](https://github.com/chandrikadeb7)

## :+1: Credits
* [https://www.pyimagesearch.com/](https://www.pyimagesearch.com/)
* [https://www.tensorflow.org/tutorials/images/transfer_learning](https://www.tensorflow.org/tutorials/images/transfer_learning)

## :handshake: Our Contributors

[CONTRIBUTORS.md](/CONTRIBUTORS.md)


## :eyes: Code of Conduct

You can find our Code of Conduct [here](/CODE_OF_CONDUCT.md).

## :eyes: License
MIT © [Chandrika Deb](https://github.com/chandrikadeb7/Face-Mask-Detection/blob/master/LICENSE)

