# Implement-Vector-Quantized-Variational-Autoencoder-VQ-VAE-for-defect-detection-in-product-images.

The VQ-VAE is a type of generative model that combines variational inference with vector quantization to learn discrete latent representations of image data. Unlike traditional autoencoders, VQ-VAE encodes input images into discrete latent codes, which improves the quality and interpretability of reconstructions, making it suitable for applications like anomaly detection.

# Architecture
![image](https://github.com/user-attachments/assets/d5b6797b-4147-48f6-834c-858f2a7fcec1)
Fig 1 shows various top level components in the architecture along with dimensions at each step. Assuming we run our model over image data, here’s some nomenclature we’ll be using going forward:
n : batch size
h: image height
w: image width
c: number of channels in the input image
d: number of channels in the hidden state

# Vector Quantization Layer
![image](https://github.com/user-attachments/assets/67fbcc4e-6c2f-4e2d-abe0-de795864b7c4)
The working of VQ layer can be explained in six steps as numbered in Fig 2:
1. Reshape: all dimensions except the last one are combined into one so that we have n*h*w vectors each of dimensionality d

2. Calculating distances: for each of the n*h*w vectors we calculate distance from each of k vectors of the embedding dictionary to obtain a matrix of shape (n*h*w, k)

3. Argmin: for each of the n*h*w vectors we find the the index of closest of the k vectors from dictionary

4. Index from dictionary: index the closest vector from the dictionary for each of n*h*w vectors

5. Reshape: convert back to shape (n, h, w, d)

6. Copying gradients: If you followed up till now you’d realize that it’s not possible to train this architecture through backpropagation as the gradient won’t flow through argmin. Hence we try to approximate by copying the gradients from z_q back to z_e. In this way we’re not actually minimizing the loss function but are still able to pass some information back for training.

# How It Works
The model is trained exclusively on defect-free (normal) product images. During training, it learns the distribution and structural patterns of these normal images. At inference time, the model attempts to reconstruct new input images. Since it has never seen defective samples, it fails to accurately reconstruct anomalous regions, resulting in higher reconstruction errors in those areas.

By computing the pixel-wise difference between the original and reconstructed image, we can highlight defective regions, making this approach highly effective for localizing and identifying defects without any need for labeled defect data.

# Key Features
> Implementation of the full VQ-VAE architecture using PyTorch
> Reconstruction error-based anomaly detection for defect identification
> Support for custom datasets of product images
> Visualization tools to compare original, reconstructed, and anomaly maps
> Can be extended for real-time industrial inspection systems

# Applications
> Quality inspection in manufacturing lines
> Anomaly detection in packaging, electronics, and mechanical parts
> Visual inspection automation where labeled defect data is limited or unavailable
> This approach offers a scalable and efficient solution to detect and localize defects, reducing human effort and enhancing inspection reliability. The project can be easily customized for different types of products by training on relevant image datasets.

# Conclusion
>The implementation of Vector Quantized Variational Autoencoder (VQ-VAE) for defect detection demonstrates the potential of unsupervised learning in industrial quality inspection tasks. By learning to reconstruct only normal (defect-free) patterns, the model effectively identifies deviations or anomalies in unseen data without the need for labeled defect samples.
> This approach not only simplifies the training process but also makes the solution scalable and adaptable to various product types and defect categories. With its strong reconstruction capabilities and efficient anomaly localization, VQ-VAE provides a powerful and interpretable framework for automated visual inspection.
> Future improvements could include integrating real-time deployment pipelines, optimizing model performance for edge devices, and combining VQ-VAE with attention mechanisms or other hybrid models for enhanced precision.
