# Unsupervised Representation Learning by Predicting Image Rotations

<ul>
	<li>
		Method: <strong>Self-Supervised</strong>
		<ul>
			<li>
				Proxy Problem: <strong>Predicting Image Rotations</strong>
			</li>
		</ul>
	</li>
</ul>

# Proxy Problem

For self-supervised representation learning, a neural network is trained to predict the rotation applied to the input image. If the function $Rot(X, \phi)$ rotates the input image $X$ by $\phi$ degrees, the function $g_{K}(X,y)$ is defined as follows:

$$g_{K}(X,y) = Rot(X, \frac{2\pi}{K} y)$$

The $g_{K}(X,y)$ function applies K different rotations to the input image $X$ and actually the $y$ value represents one of these $K$ rotations:

$$ y = \{0,1,\dots,K-1\}$$

In the proxy problem, we are looking for a neural network ($F$) that can find the amount of rotation. Since there are K different rotations, we have a K-class classification problem. Therefore, the neural network $F$ is trained to find the probability of each of the K rotations. The probablity of $\frac{2\pi}{K} y^{\star}$ rotation is defined as follows:

$$ F^{y^{\star}}(\hat{x}|\theta)$$

In the above expression, $\theta$ is the network parameters.

# Loss Function
The loss value for input $X_{i}$ is calculated as follows:

$$loss(X_{i}, \theta) = -\frac{1}{K} \sum_{y=1}^{K}{log( F^{y}(g(X_{i},y)|\theta) )}$$

Therefore, the totall loss value for dataset $ D=\{X_{i} | 1 \leq i \leq N \}$ is:

$$ Loss(D, \theta) = \frac{1}{N} \sum_{i=1}^{N}{loss(X_{i},\theta)} $$

# Objective
In this problem, the objective is to reduce the totall loss value, or in other words, the goal is to increase the probability of correctly detecting a rotation.

$$ \min_{\theta} Loss(D, \theta) $$

<table style="text-align: center margin-left: auto; margin-right: auto; text-align: center" border=0 align=center>
    <tbody style="text-align: center margin-left: auto; margin-right: auto; text-align: center" border=0 align=center>
        <tr>
            <td>
                <img src="./plots/paper/Rotation_Prediction.png" alt="Sample from Cifar-10 Testing Data" style="width: 50rem"/>
            </td>
        </tr>
        <tr>
        	<td>
        		Unsupervised Representation Learning by Predicting Image Rotations
        	</td>
        </tr>
    </tbody>
</table>

# Dataset

<ul>
	<li>
		<a href="https://www.cs.toronto.edu/~kriz/cifar.html">Cifar-10</a>
		<ul>
			<li>
				Training vs Testing Distribution:
				<table style="text-align: center margin-left: auto; margin-right: auto; text-align: center" border=0 align=center>
				    <tbody>
				        <tr>
				            <td>
				                <img src="./plots/EDA/train_vs_test.png" alt="Training vs Testing Distribution" style="width: 40rem"/>
				            </td>
				        </tr>
				    </tbody>
				</table>
			</li>
			<li>
				Some Samples from training and testing data:
			</li>
		</ul>
	</li>
</ul>

<table style="text-align: center margin-left: auto; margin-right: auto; text-align: center" border=0 align=center>
    <tbody style="text-align: center margin-left: auto; margin-right: auto; text-align: center" border=0 align=center>
        <tr>
            <td>
                <img src="./plots/EDA/train_label_images.png" alt="Sample from Cifar-10 Training Data" style="width: 40rem"/>
            </td>
        </tr>
        <tr>
        	<td>
        		A sample from Training Data
        	</td>
        </tr>
    </tbody>
</table>

<table style="text-align: center margin-left: auto; margin-right: auto; text-align: center" border=0 align=center>
    <tbody style="text-align: center margin-left: auto; margin-right: auto; text-align: center" border=0 align=center>
        <tr>
            <td>
                <img src="./plots/EDA/test_label_images.png" alt="Sample from Cifar-10 Testing Data" style="width: 40rem"/>
            </td>
        </tr>
        <tr>
        	<td>
        		A sample from Testing Data
        	</td>
        </tr>
    </tbody>
</table>
