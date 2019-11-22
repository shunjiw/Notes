### Deep Neural Network (DNN)

### Convolutional Neural Network (CNN)

#### Computer vision

Large image needs billions of parameters which is infeasible for original NN.

* Edge Detection:
  - Vertical and horizontal edges
  - Vertical filter [[1, 0, -1], [1, 0, -1], [1, 0, -1]] 
  - Horizontal filter [1, 1, 1], [0,0,0], [-1,-1,-1]]
  - It’s the building block of CNN
* Choices of filters:
  - Filter size is f and f is always odd, like 3, 5, 7.
  - Vertical filters: [[1,0,-1], [2,0,-2], [1,0,-1]]
  - Treat the filter as parameters to learn: [[w1,w2,w3], [w4,w5,w6], …]
* Padding:
  - Downsides of filtering: images will shrink, a pixel will be fitted many times.
  - Pad the image from 6*6 to 8*8 (p = padding = 1) with fill in all 0s.
  - Choice of padding p: 
    - ‘Valid’ means no padding: n*n * f*f = n-f+1 * n-f+1
    - ‘Same’ means p = (f-1)/2
  - It allows you to use a CONV layer without necessarily shrinking the height and width of the volumes. 
  - It helps us keep more of the information at the border of an image.
* Strided convolution:
  - When s = stride = 2, the filter moves s steps one time
  - So, when padding = p and stride = s, the output side is floor of (n+2p-f)/s + 1.
* Convolutions on RGB images:
  - RGB images are (n, n, 3), named (height, weight, # channels)
  - So, filters are (f, f, 3)
  - Perform element-wise multiply, so end up with a matrix output.
  - Example, if you care about the red vertical filter, you let the red filter as [[1, 0, -1], [1, 0, -1], [1, 0, -1]] and the green and blue ones are all zeros.
* Multiple filters:
  - (6,6,3)*(3,3,3)_3 ->(4,4)_3 stack them as (4,4,3) 
  - (n,n,nc) * (f,f,nc) -> (n-f+1, n-f+1, nc’), nc is # channels and nc’ is # filters.
* One layer of CNN:
  -	(6,6,3) *(3,3,3)_3 -> relu ((4,4) + b) -> (4,4)_3 -> stack them as (4,4,3) 
  -	Number of parameters in one layer is 280, if we have 10 filters.
  -	A good property is no matter how large of image (it can be 1000*1000 pixels), the number of parameters in one layer is fixed.
* Notation for CNN:
  -	the f[l] is the filter size.
  -	p[l] is the amount of padding, to make the output have the same shape of the input.
  -	S[l] is the amount of stride
  -	N_c[l] is the number of filters
  -	Input: n_h[l-1] * n_w[l-1] *n_c[l-1].
  -	Output: n_h[l] * n_w[l] *n_c[l].
  -	Each filter: f[l] * f[l] * n_c[l-1].
  -	Activations: n_h[l] * n_w[l] * n_c[l]. 
  -	Weights: f[l] * f[l] * n_c[l-1] * n_c[l]. 
- ConvNet:
  +	the 39,39,3 -> 37,37,10 -> 17,17,20 -> 7,7,40 -> 1940 units -> y_hat.
  +	the n_h and n_w are decreasing and n_c is increasing.
  +	Types of layers in CNN:
		-	Convolution (CONV): take the element-wise multiply over the region
		-	Pooling (POOL)
		-	Fully connected (FC)
*	Pooling layer:
  -	Max pooling: take the maximum over the region.
    -	It’s done individually for each channel
    - There is no parameter to learn when you fix f and s
  -	Average Pooling: take the average over the region.
  -	Hyperparameters: f = 2, s = 2, max or average pooling
  -	P = 0
*	Fully connected layer:
  -	It’s just the ordinary layer in DNN, like a matrix of weights (400, 120)
*	An example of CNN:
  -	CONV1 -> Pool -> CON2 -> Pool -> FC -> FC -> FC -> maxsoft
	 
*	Why convolutions?
  -	Parameters sharing: a filter detection which is useful for one part is also useful for another part. (Compare to fully connected layer, the number of parameters is very small)
  -	Sparsity of connections: each output value depends on a small number of inputs. It avoids overfitting and captures translation invariance.
*	Backward pass:
  -	Even though a pooling layer has no parameters for backprop to update, you still need to backpropagation the gradient through the pooling layer in order to compute gradients for layers that came before the pooling layer.
  -	Why do we keep track of the position of the max? It's because this is the input value that ultimately influenced the output, and therefore the cost. Backprop is computing gradients with respect to the cost, so anything that influences the ultimate cost should have a non-zero gradient. So, backprop will "propagate" the gradient back to this particular input value that had influenced the cost.

#### Case Study
*	LeNet-5:
  -	32*32*1 -- conv layer -- 28*28*6 – average pool – 14*14*6 -- conv layer -- 10*10*16 – average pool – 5*5*16 – FC – 120 – FC – 84 – softmax – y_hat.
  -	60k parameters
  -	N_H, n_w decreases but n_c increases.
*	AlexNet:
  -	Similar to LeNet-5, but much bigger, around 60m parameters
  -	Use Relu
*	VGG-16:
  -	It uses unified layer to make the architecture simple. The convolutional layer is 3*3 filter, s = 1, same and the max pool is 2*2, s = 2.
  -	N_c doubles every time.
*	ResNets (Residual Networks)
  -	Residual block
    -	Mian path:  a[l] – linear – Relu – a[l+1] - linear - Relu – a[l+2]
    -	Shortcut: a[l+2] = Relu(z[l+2] + a[l])
  -	ResNets is a plant network but each two layers are a residual block.
  -	The advantage of residual block is avoiding exploring or vanishing gradient and making the training error keep decreasing as the number of layers.
  -	But for ordinary plant network, the training error may increase as the number of layer because the optimization algorithm has a hard time training deep NN.
*	Why residual network works?
  -	a[l+2] = Relu(w[l+2]a[l+1] + b[l+2] + a[l]) 
  -	If w[l+2] = b[l+2] = 0, a[l+2] = a[l] because a[l] are positive. 
  -	So, it’s easy for residual network to learn identity function which means adding two more hidden layers won’t hurt the performance. (perform at least as good as a[l])
  -	One restriction of residual network is that a[l] must have the same dimension as z[l+2]. So, it’s also the reason why there several layers have the same dimensions in some NN.
  -	To correct the disagreement between a[l] and a[l+2], use another weights w[l+2]’ like a[l+2] = Relu(z[l+2] + w[l+2]’a[l])
*	1 by 1 convolution:
  -	6 * 6 * 32 image multiples 1 * 1 * 32 which turns out a 6 * 6 matrix.
  -	It likes a full connected layer applied to each slice for all positions in (height, weight)
  -	It’s a way to keep n_h and n_w but shrink the dimension of channels.
*	Inception layer:
  -	It’s a united layer which stack up different kinds of filters, called inception layer.
  -	For example, the input layer is 28*28*192 and is filtered by 1*1 filter, which ends up with 28*28*64 volume. it is filtered by 3*3 filter, which ends up with 28*28*128 volume. … Keep all the volume the same shape so they can be stacked up.
*	‘Bottleneck’ layer:
  -	It’s a layer between two existing layers. By 1*1 filter, it can reduce the computational cost dramatically compared to a 5*5 convolutional layer.	 
  -	So, the inception network is built by above inception module.
*	Workflow of Conv NN:
  -	Use open-source implementation from Github.
  -	Transfer learning from pre-trained model
    -	Download the code and weights
    -	Freeze the weights in middle and change the output layer.
    -	Or, we can compute the last feature vectors for your data and save it for training further layers.
    -	If you have more data, the number of frozen layers decrease.
*	Data augmentation:
  -	For computer vision, more data is always better.
  -	Ways of data augmentation: mirroring, random cropping, rotation, color shifting.
*	Data vs hand-engineering:
  -	Little data needs more hand-engineering.
  -	A lot of data can afford simple algo and less hand-engineering.
  -	Two sources of knowledges:
    -	Labeled data
    -	Hand engineering features/ network architecture/ others
*	Tips for doing well on benchmarks/winning competitions:
  -	Ensemble several trained NN and average the y_hat.
  -	10 crops at test time. Randomly select 10-crop of the test image and average of the y hat of these crops.
  -	Use open source code:
    -	Use architectures of networks published in the literature.
    -	Use open source implementation if possible.
    -	Use pre-trained models and fine-tune on your datasets.

Week 3. Detection Algorithms
*	Object Localization
  -	Classification with localizations: Not only classify which kind of object is in the picture, but also predict the location of the object like (bx, by, bw, bh).
  -	Target label y = [ Pc, bx, by, bh, bw, c1, c2, c3, …] where Pc is the probability of whether there is an object, c1, …, ck is the probability of object is ci. 
  -	So, if there is a car in the picture, y = [1, 0.5, 0.3, 0.1, 0.2, 0, 1, 0, …]. If there is nothing in the picture, y = [0, don’t care, …]

### Recursive Neural Network (RNN)
