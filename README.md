## Handwritten Digit Images Generation using Generative Adversarial Network (GAN)

This is the final project of ECE 417 Multimedia Processing SP21

### Reference

**Pytorch's DCGAN Tutorial**

URL: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

**FID evaluation**

URL: https://github.com/IsChristina/pytorch-fid

### Training

To train the DCGAN, please run following command in console

```python
python dcgan.py
```

If it is the first running, it would download MNIST dataset into data folder.

### Evaluation

To evaluate the model, please 

- Get fake images. Use **loader.py** to load the model which is trained and saved by **dcgan.py** (path to model file should be set in **loader.py**), then, generate fake images.

- Get real images. Please comment parts of the codes in **loader.py** then uncomment other parts.

After doing these, we have two folders of real and fake images respectively. Please go to the folder ***eval***.

There are two folders in it, one is FID, another is MISSSIM.

To run FID, please use the following command in console opened in FID folder

```bash
./fid_score.py path/to/dataset1 path/to/dataset2
```

To run MISSSIM, please set the proper path in **eval_misssim.py** then run it. If the library is not installed, please use following command to install it:

```python
pip install pytorch_msssim
```

### Results

More results and trained models can be seen in ***results*** folder. Here are parts of them.

#### **50 epochs, batch size 128, one-sided smoothing (true label set to 0.9)**

**Animation**

![](https://github.com/TaKeTube/MNIST-GAN/blob/main/results/animation.gif?raw=true)

**Compare with real image**

![](https://github.com/TaKeTube/MNIST-GAN/blob/main/results/50%20epoch%20bs%20128%20label%200.9/results/compare.png?raw=true)

**Loss  during training** 

![](https://github.com/TaKeTube/MNIST-GAN/blob/main/results/50%20epoch%20bs%20128%20label%200.9/results/loss.png?raw=true)

**D(x) & D(G(z)) during training**

![](https://github.com/TaKeTube/MNIST-GAN/blob/main/results/50%20epoch%20bs%20128%20label%200.9/results/probability.png?raw=true)

#### A Failed training (200 epochs, batch size 256)

**Animation**

![](https://github.com/TaKeTube/MNIST-GAN/blob/main/results/fail-animation.gif?raw=true)

You can see mode collapse in this process.

**Loss  during training** 

![](https://github.com/TaKeTube/MNIST-GAN/blob/main/results/200%20epoch/results/fail-loss.png?raw=true)

**D(x) & D(G(z)) during training**

![](https://github.com/TaKeTube/MNIST-GAN/blob/main/results/200%20epoch/results/probability.png?raw=true)