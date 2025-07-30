# RF-BASED-DRONE-SURVEILLANCE-SYSTEM-USING-WAVELET-SCATTERING-TRANSFORM

This is my graduation thesis, and it focuses on the use of Wavelet Scattering Transform (WST) for RF signal feature extraction. These features are then used as inputs for classic Machine Learning algorithms: Random Forest and Tree Bagging for Drone classification. I also propose an optimized solution for a real-time algorithm implemented in hardware. The dataset I use is DroneRF, as documented in "*DroneRF dataset: A dataset of drones for RF-based detection, classification and identification*". Details about Wavelet Scattering you can find in "*Deep scattering spectrum*", "*“Understanding deep convolutional networks*", "*“Wavelet Scattering Transform. Mathematical Analysis and Applications to VIRGO Gravitational Waves Data*"

I will outline the main parts of my thesis:

1. The dataset is divided into two types of RF signals: 0–40 MHz and 40–80 MHz.
I apply WST and ML algorithms to each type and use a decision-making system to determine the final prediction.

2. RF signals in the dataset are IF signals with 10⁶ samples.
To balance global and local features, I apply WST to chunks of the signal (each of length 2.5e6). After transforming each chunk, I combine the results to form the feature vector for one signal.

3. There are three tasks: Drone detection, Drone classification, and Drone operation classification.
The first two tasks have large enough datasets to use Random Forest as the classifier. However, the drone operation classification task has limited data due to the higher number of classes. Therefore, I use Tree Bagging to maximize the information that can be learned.
(The reason I use Random Forest is that I tested several algorithms, and it performed the best.)

4. We have two models. To decide which classification is correct, I use a confidence score.
The idea is simple: both 0–40 MHz and 40–80 MHz signals carry information about the drone. One band may capture features that the other cannot. The prediction with the higher confidence score will be chosen as the final decision.
However, this approach may face the over-confidence problem, which I address by applying weights to each score.

5. The model’s complexity is mostly caused by WST.
On a GTX 1660Ti, it takes around 8 seconds to complete, which is relatively long since drone classification often requires real-time prediction.
I believe this is only a soft constraint — it doesn’t necessarily need to be 1ms or faster. Less than 1 second should be sufficient.
My proposed solution is to design a WST algorithm using lookup tables that store pre-calculated values of FFT, the mother wavelet function, etc.
One advantage of WST is that many experiments show we can use constant parameters. This is important because hardware algorithms need static, not dynamic, parameters.

## Note
For classification: classification.m, signal_split.m, operation_classification.
