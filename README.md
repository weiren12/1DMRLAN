# Real-Time Rotating Machinery Fault Diagnosis System Based on Multiscale Convolution Attention Network and Vibration Data (MRLAN)
This is the algorithm specifically constructed for real-time intelligent fault diagnosis (RTIFD) based on the proposed multiscale residual LSTM attention network (MRLAN). This code is written by Sia Wei Ren from Universiti Teknologi Malaysia (UTM) under the departments of the Institute of Noise and Vibration (IKG) and the Faculty of Mechanical Engineering (FME).

# Multiscale Residual LSTM Attention Network (MRLAN)
Code Link
<img width="6750" height="9792" alt="MRLAN_Structure" src="https://github.com/user-attachments/assets/88e42aad-05b6-4c5c-bd51-e144c60a9fa0" />

# List of Comparison Models
These models are only available in TensorFlow. Some models are built in TensorFlow and have subtle differences if built in PyTorch.
## List of Self-rebuilded online published models for comparison
| No. | Model | Article Name |
| --- | --- |  --- |
| 1. | CNN-BiLSTM | [An optimized CNN-BiLSTM network for bearing fault diagnosis under multiple working conditions with limited training samples](https://doi.org/10.1016/j.neucom.2024.127284) |
| 2. | MSC-MpResCNN | [A novel convolutional neural network with multiscale cascade midpoint residual for fault diagnosis of rolling bearings](https://doi.org/10.1016/j.neucom.2022.07.022) |
| 3. | MSCNN-LSTM | [Anti‐noise diesel engine misfire diagnosis using a multi‐scale CNN‐LSTM neural network with denoising module](https://doi.org/10.1049/cit2.12170) |
| 4. | MCL-DRL | [An anti-noise fault diagnosis approach for rolling bearings based on multiscale CNN-LSTM and a deep residual learning model](https://doi.org/10.1088/1361-6501/acb074) |
| 5. | AMPCNN | [A Fault Diagnosis Method of Rotor System Based on Parallel Convolutional Neural Network Architecture with Attention Mechanism](https://doi.org/10.1007/s11265-023-01846-y) |

## List of Base Models
Names of Models in the table below are the same as the names in the function.
| No. | Models  | Description |
| --- | --- |  --- |
| 1. | `CNN` | Basic CNN |
| 2. | `CNNRes` | CNN + Residual Block |
| 3. | `CNNLSTM` | CNN + LSTM |
| 4. | `CNNMHA` | CNN + Multiheaded Self-Attention  |

# Env Version and Software
* Software: VS Code + LabVIEW 2025
* Python: 3.10
* PyTorch Version: 2.9.1 
* Tensorflow Version: 2.18.0
* Scikit-learn 1.5.2

# Contact
Name: Sia Wei Ren\
Email: [weiren1212@gmail.com](weiren1212@gmail.com)


If you find the code or information is useful, we sincerely appreciate you in citing our article!
